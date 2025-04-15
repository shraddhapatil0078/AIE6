from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai.embeddings import OpenAIEmbeddings
from enhanced_rag_graph import EnhancedRAGGraph

def setup_rag_system():
    # Initialize the LLM
    llm = ChatOpenAI(model="gpt-4")
    
    # Initialize the embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Initialize Qdrant client and vector store
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="test_collection",
        embedding=embedding_model
    )
    
    # Create a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Create the chat prompt template
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("human", """Context: {context}
        
        Question: {query}
        
        Please provide a detailed answer based on the context. If the context doesn't contain relevant information, say "I don't know".""")
    ])
    
    # Initialize the enhanced RAG graph
    rag_graph = EnhancedRAGGraph(llm, retriever, chat_prompt)
    graph = rag_graph.build_graph()
    
    return graph

def main():
    # Setup the RAG system
    graph = setup_rag_system()
    
    # Example questions
    questions = [
        "What is the capital of France?",
        "What are the main features of LCEL?",
        "How does LangGraph work?"
    ]
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: {question}")
        result = graph.invoke({"question": question})
        print(f"Response: {result['response']}")
        print(f"Relevance Check: {result['relevance_check']}")
        print(f"Fact Check: {result['fact_check']}")

if __name__ == "__main__":
    main() 