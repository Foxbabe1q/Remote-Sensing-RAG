"""
LangGraph-based RAG workflow for question answering.
Implements a state machine for the retrieval-augmented generation pipeline.
"""

import logging
import base64
from typing import List, Dict, Any, TypedDict, Annotated, Optional, BinaryIO
from operator import add

from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


# Define the state structure for the graph
class RAGState(TypedDict):
    """
    State object that flows through the RAG graph.
    """
    question: str  # User's question
    retrieved_documents: List[Document]  # Documents retrieved from vector store
    context: str  # Formatted context from retrieved documents
    answer: str  # Generated answer
    metadata: Dict[str, Any]  # Additional metadata
    errors: Annotated[List[str], add]  # List of errors encountered


class RAGGraph:
    """
    LangGraph-based RAG (Retrieval-Augmented Generation) workflow.
    Orchestrates document retrieval and answer generation.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        top_k: int = 3,
        api_key: str = None
    ):
        """
        Initialize the RAG graph.
        
        Args:
            vector_store_manager: VectorStoreManager instance for retrieval
            llm_model: Name of the LLM model to use
            temperature: Temperature for LLM generation
            top_k: Number of documents to retrieve
            api_key: OpenAI API key
        """
        self.vector_store_manager = vector_store_manager
        self.llm_model = llm_model
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=api_key
        )
        
        # Create the prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(
            f"RAGGraph initialized with model={llm_model}, "
            f"temperature={temperature}, top_k={top_k}"
        )
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for answer generation.
        
        Returns:
            ChatPromptTemplate for RAG
        """
        template = """You are an expert assistant specializing in remote sensing and image segmentation research.
Your task is to answer questions based on the provided research paper excerpts.

Context from research papers:
{context}

Question: {question}

Instructions:
1. Answer the question based ONLY on the information provided in the context above.
2. If the context doesn't contain enough information to answer the question, say so clearly.
3. Cite specific details from the papers when possible.
4. If multiple papers discuss the topic, synthesize the information.
5. Be concise but comprehensive in your answer.
6. Use technical terminology appropriately.
7. For mathematical formulas, use LaTeX notation with:
   - Inline formulas: $formula$
   - Block formulas: $$formula$$
   Example: The Panoptic Quality is calculated as $PQ = SQ \\times RQ$

Answer:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve relevant documents based on the question.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with retrieved documents
        """
        question = state["question"]
        
        logger.info(f"Retrieving documents for question: '{question[:100]}...'")
        
        try:
            # Perform similarity search
            documents = self.vector_store_manager.similarity_search(
                query=question,
                k=self.top_k
            )
            
            state["retrieved_documents"] = documents
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["num_retrieved"] = len(documents)
            
            logger.info(f"Retrieved {len(documents)} documents")
            
        except Exception as e:
            error_msg = f"Error during document retrieval: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", []) + [error_msg]
            state["retrieved_documents"] = []
        
        return state
    
    def _format_context(self, state: RAGState) -> RAGState:
        """
        Format retrieved documents into context string.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with formatted context
        """
        documents = state.get("retrieved_documents", [])
        
        if not documents:
            state["context"] = "No relevant documents found."
            return state
        
        logger.info("Formatting context from retrieved documents")
        
        # Format each document with metadata
        formatted_docs = []
        for idx, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            
            formatted_doc = f"""
Document {idx} (Source: {source}, Page: {page}):
{content}
---"""
            formatted_docs.append(formatted_doc)
        
        state["context"] = "\n\n".join(formatted_docs)
        
        return state
    
    def _generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using LLM based on context and question.
        
        Args:
            state: Current RAG state
            
        Returns:
            Updated state with generated answer
        """
        question = state["question"]
        context = state.get("context", "")
        
        logger.info("Generating answer with LLM")
        
        try:
            # Create the prompt
            prompt = self.prompt_template.format_messages(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.llm.invoke(prompt)
            state["answer"] = response.content
            
            # Update metadata
            state["metadata"] = state.get("metadata", {})
            state["metadata"]["llm_model"] = self.llm_model
            state["metadata"]["answer_length"] = len(response.content)
            
            logger.info("Answer generated successfully")
            
        except Exception as e:
            error_msg = f"Error during answer generation: {str(e)}"
            logger.error(error_msg)
            state["errors"] = state.get("errors", []) + [error_msg]
            state["answer"] = "Error: Unable to generate answer."
        
        return state
    
    def _should_continue(self, state: RAGState) -> str:
        """
        Determine if the graph should continue or end.
        
        Args:
            state: Current RAG state
            
        Returns:
            Next node name or END
        """
        errors = state.get("errors", [])
        
        if errors:
            logger.warning(f"Errors detected: {errors}")
            return END
        
        return END
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        logger.info("Building RAG graph workflow")
        
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_documents)
        workflow.add_node("format_context", self._format_context)
        workflow.add_node("generate", self._generate_answer)
        
        # Define the flow
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "format_context")
        workflow.add_edge("format_context", "generate")
        workflow.add_edge("generate", END)
        
        # Compile the graph
        compiled_graph = workflow.compile()
        
        logger.info("RAG graph workflow built successfully")
        return compiled_graph
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline for a given question.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: '{question[:100]}...'")
        
        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "retrieved_documents": [],
            "context": "",
            "answer": "",
            "metadata": {},
            "errors": []
        }
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Extract results
            result = {
                "question": question,
                "answer": final_state.get("answer", "No answer generated."),
                "sources": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "content_preview": doc.page_content[:200] + "..."
                    }
                    for doc in final_state.get("retrieved_documents", [])
                ],
                "metadata": final_state.get("metadata", {}),
                "errors": final_state.get("errors", [])
            }
            
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            error_msg = f"Error executing RAG pipeline: {str(e)}"
            logger.error(error_msg)
            return {
                "question": question,
                "answer": "Error processing query.",
                "sources": [],
                "metadata": {},
                "errors": [error_msg]
            }
    
    def query_with_image(self, question: str, image_file: BinaryIO) -> Dict[str, Any]:
        """
        Execute the RAG pipeline with both text question and image input.
        Uses multimodal capabilities of GPT-4o/GPT-4o-mini.
        
        Args:
            question: User's question
            image_file: Uploaded image file object
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing multimodal query: '{question[:100]}...' with image")
        
        try:
            # First, do RAG retrieval based on text question
            logger.info("Step 1: Retrieving relevant documents")
            documents = self.vector_store_manager.similarity_search(
                query=question,
                k=self.top_k
            )
            
            # Format context from retrieved documents
            context = ""
            if documents:
                formatted_docs = []
                for idx, doc in enumerate(documents, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    content = doc.page_content
                    formatted_doc = f"Document {idx} (Source: {source}, Page: {page}):\n{content}\n---"
                    formatted_docs.append(formatted_doc)
                context = "\n\n".join(formatted_docs)
            else:
                context = "No relevant documents found in the database."
            
            # Encode image to base64
            logger.info("Step 2: Encoding image")
            image_bytes = image_file.read()
            image_file.seek(0)  # Reset file pointer
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            # Determine image type
            image_type = "image/jpeg"
            if hasattr(image_file, 'type'):
                image_type = image_file.type
            elif hasattr(image_file, 'name'):
                if image_file.name.lower().endswith('.png'):
                    image_type = "image/png"
                elif image_file.name.lower().endswith('.webp'):
                    image_type = "image/webp"
            
            # Create multimodal message with image and context
            logger.info("Step 3: Generating answer with vision model")
            
            prompt_text = f"""You are an expert assistant specializing in remote sensing and image segmentation research.

Context from research papers:
{context}

The user has uploaded an image and asked the following question:
{question}

Instructions:
1. Analyze the uploaded image carefully.
2. Answer the question based on BOTH the image content AND the context from research papers.
3. If the image contains diagrams, charts, or figures, describe and explain them.
4. Relate the image content to concepts from the research papers when relevant.
5. If the context doesn't contain relevant information, focus on analyzing the image.
6. Be detailed and technical in your analysis.
7. For mathematical formulas, use LaTeX notation with:
   - Inline formulas: $formula$
   - Block formulas: $$formula$$
   Example: The Panoptic Quality is $PQ = SQ \\times RQ$

Answer:"""
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{base64_image}"
                        }
                    }
                ]
            )
            
            # Generate response with vision-enabled model
            response = self.llm.invoke([message])
            answer = response.content
            
            logger.info("Multimodal query processed successfully")
            
            return {
                "question": question,
                "answer": answer,
                "sources": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "page": doc.metadata.get("page", "N/A"),
                        "content_preview": doc.page_content[:200] + "..."
                    }
                    for doc in documents
                ],
                "metadata": {
                    "num_retrieved": len(documents),
                    "llm_model": self.llm_model,
                    "answer_length": len(answer),
                    "multimodal": True,
                    "has_image": True
                },
                "errors": []
            }
            
        except Exception as e:
            error_msg = f"Error processing multimodal query: {str(e)}"
            logger.error(error_msg)
            return {
                "question": question,
                "answer": "Error processing query with image. Please try again.",
                "sources": [],
                "metadata": {"multimodal": True, "has_image": True},
                "errors": [error_msg]
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple questions in batch.
        
        Args:
            questions: List of questions
            
        Returns:
            List of results for each question
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        
        logger.info("Batch processing completed")
        return results

