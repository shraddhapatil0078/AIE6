from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

class RAGComponents:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        
    def check_relevance(self, query: str, documents: List[Document]) -> Dict[str, Any]:
        """Check if retrieved documents are relevant to the query."""
        relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that evaluates the relevance of documents to a query."),
            ("human", """Given the following query and documents, determine if the documents are relevant.
            
            Query: {query}
            
            Documents:
            {documents}
            
            Return a JSON object with two fields:
            1. 'is_relevant': boolean indicating if the documents are relevant
            2. 'reason': string explaining why the documents are or aren't relevant
            """)
        ])
        
        chain = relevance_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": query,
            "documents": "\n\n".join([doc.page_content for doc in documents])
        })
        
        # Parse the result into a dictionary
        # Note: In a production environment, you'd want to use a proper JSON parser
        is_relevant = "true" in result.lower()
        return {"is_relevant": is_relevant, "reason": result}

    def rewrite_query(self, query: str, reason: str) -> str:
        """Rewrite the query to improve retrieval results."""
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that improves queries for better document retrieval."),
            ("human", """The following query did not retrieve relevant documents. 
            Reason: {reason}
            
            Original query: {query}
            
            Please rewrite the query to be more specific and likely to retrieve relevant documents.
            """)
        ])
        
        chain = rewrite_prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "reason": reason})

    def fact_check(self, query: str, response: str, documents: List[Document]) -> Dict[str, Any]:
        """Check if the generated response is factually accurate based on the documents."""
        fact_check_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that fact-checks responses against provided documents."),
            ("human", """Given the following query, response, and source documents, check if the response is factually accurate.
            
            Query: {query}
            
            Response: {response}
            
            Source Documents:
            {documents}
            
            Return a JSON object with two fields:
            1. 'is_accurate': boolean indicating if the response is factually accurate
            2. 'corrections': list of any factual inaccuracies found
            """)
        ])
        
        chain = fact_check_prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "query": query,
            "response": response,
            "documents": "\n\n".join([doc.page_content for doc in documents])
        })
        
        # Parse the result into a dictionary
        # Note: In a production environment, you'd want to use a proper JSON parser
        is_accurate = "true" in result.lower()
        return {"is_accurate": is_accurate, "corrections": result}

    def refine_response(self, query: str, response: str, corrections: str, documents: List[Document]) -> str:
        """Refine the response to correct any inaccuracies."""
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that corrects factual inaccuracies in responses."),
            ("human", """Given the following query, original response, corrections, and source documents, 
            please provide a corrected response that addresses all factual inaccuracies.
            
            Query: {query}
            
            Original Response: {response}
            
            Corrections Needed: {corrections}
            
            Source Documents:
            {documents}
            
            Please provide a corrected response that is factually accurate based on the source documents.
            """)
        ])
        
        chain = refine_prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "query": query,
            "response": response,
            "corrections": corrections,
            "documents": "\n\n".join([doc.page_content for doc in documents])
        }) 