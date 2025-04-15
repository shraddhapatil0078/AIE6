from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from rag_components import RAGComponents

class State(TypedDict):
    question: str
    context: List[Document]
    response: str
    relevance_check: dict
    fact_check: dict

class EnhancedRAGGraph:
    def __init__(self, llm, retriever, chat_prompt):
        self.llm = llm
        self.retriever = retriever
        self.chat_prompt = chat_prompt
        self.components = RAGComponents(llm)
        
    def retrieve(self, state: State) -> State:
        """Retrieve documents based on the question."""
        retrieved_docs = self.retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    
    def check_relevance(self, state: State) -> State:
        """Check if retrieved documents are relevant."""
        relevance_result = self.components.check_relevance(
            state["question"], 
            state["context"]
        )
        return {"relevance_check": relevance_result}
    
    def rewrite_query(self, state: State) -> State:
        """Rewrite the query if documents are not relevant."""
        new_query = self.components.rewrite_query(
            state["question"],
            state["relevance_check"]["reason"]
        )
        return {"question": new_query}
    
    def generate(self, state: State) -> State:
        """Generate a response using the retrieved context."""
        generation_chain = self.chat_prompt | self.llm | StrOutputParser()
        response = generation_chain.invoke({
            "query": state["question"],
            "context": state["context"]
        })
        return {"response": response}
    
    def fact_check(self, state: State) -> State:
        """Check if the generated response is factually accurate."""
        fact_check_result = self.components.fact_check(
            state["question"],
            state["response"],
            state["context"]
        )
        return {"fact_check": fact_check_result}
    
    def refine_response(self, state: State) -> State:
        """Refine the response if it contains inaccuracies."""
        refined_response = self.components.refine_response(
            state["question"],
            state["response"],
            state["fact_check"]["corrections"],
            state["context"]
        )
        return {"response": refined_response}
    
    def build_graph(self) -> StateGraph:
        """Build the enhanced RAG graph."""
        # Initialize the graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("check_relevance", self.check_relevance)
        graph_builder.add_node("rewrite_query", self.rewrite_query)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_node("fact_check", self.fact_check)
        graph_builder.add_node("refine_response", self.refine_response)
        
        # Add edges
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "check_relevance")
        
        # Add conditional edges
        graph_builder.add_conditional_edges(
            "check_relevance",
            lambda x: "generate" if x["relevance_check"]["is_relevant"] else "rewrite_query",
            {
                "generate": "generate",
                "rewrite_query": "rewrite_query"
            }
        )
        
        graph_builder.add_edge("rewrite_query", "retrieve")
        graph_builder.add_edge("generate", "fact_check")
        
        # Add conditional edges for fact checking
        graph_builder.add_conditional_edges(
            "fact_check",
            lambda x: END if x["fact_check"]["is_accurate"] else "refine_response",
            {
                END: END,
                "refine_response": "refine_response"
            }
        )
        
        graph_builder.add_edge("refine_response", END)
        
        return graph_builder.compile() 