"""
Streamlit Web Interface for Remote Sensing Papers RAG Q&A System.

A user-friendly web application for querying research papers about
remote sensing and image segmentation using RAG technology.
"""

import streamlit as st
import sys
from pathlib import Path
import time
from datetime import datetime
import os

# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Import project modules
from config import get_settings, get_vectorstore_path
from src.vector_store import VectorStoreManager
from src.rag_graph import RAGGraph
from src.document_processor import DocumentProcessor


# Page configuration
st.set_page_config(
    page_title="Remote Sensing Papers Q&A",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        padding-bottom: 2rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .source-card {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #E0E0E0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """
    Initialize the RAG system with caching.
    This runs only once and caches the result.
    
    Returns:
        Tuple of (settings, vector_manager, rag_graph, success, error_message)
    """
    try:
        # Verify API key is loaded
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key or api_key == 'your-openai-api-key-here':
            return None, None, None, False, "OpenAI API key not configured. Please check your .env file."
        
        # Load settings
        settings = get_settings()
        vectorstore_path = get_vectorstore_path(settings)
        
        # Check if vector store exists
        if not vectorstore_path.exists():
            return None, None, None, False, "Vector store not found. Please run indexing first."
        
        # Initialize vector store manager
        vector_manager = VectorStoreManager(
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key
        )
        
        # Load vector store
        vector_manager.load_vector_store(vectorstore_path)
        
        # Initialize RAG graph
        rag_graph = RAGGraph(
            vector_store_manager=vector_manager,
            llm_model=settings.llm_model,
            temperature=settings.temperature,
            top_k=settings.top_k_results,
            api_key=settings.openai_api_key
        )
        
        return settings, vector_manager, rag_graph, True, None
        
    except Exception as e:
        return None, None, None, False, str(e)


def render_header():
    """Render the application header."""
    st.markdown('<div class="main-header">🛰️ Remote Sensing Papers Q&A System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by RAG, LangChain & LangGraph</div>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/satellite.png", width=100)
        
        st.markdown("### 📚 Navigation")
        page = st.radio(
            "Choose a page:",
            ["🏠 Home", "💬 Q&A Interface", "📊 System Stats", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Query Settings")
        top_k = st.slider(
            "Number of documents to retrieve:",
            min_value=1,
            max_value=10,
            value=3,
            help="More documents provide more context but may slow down responses"
        )
        
        temperature = st.slider(
            "Response creativity:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more factual, Higher = more creative"
        )
        
        st.markdown("---")
        
        st.markdown("### 📖 Quick Links")
        st.markdown("- [Documentation](https://github.com)")
        st.markdown("- [Report Issue](https://github.com)")
        st.markdown("- [GitHub Repo](https://github.com)")
        
        st.markdown("---")
        
        st.markdown("### 💡 Tips")
        st.info(
            "Ask specific questions for best results!\n\n"
            "**Text Examples:**\n"
            "- What is panoptic segmentation?\n"
            "- How is PQ calculated?\n"
            "- What datasets are available?\n\n"
            "**🆕 Multimodal Examples:**\n"
            "- Upload a diagram: 'Explain this architecture'\n"
            "- Upload results: 'Compare these metrics'\n"
            "- Upload equations: 'What does this formula mean?'"
        )
        
        return page, top_k, temperature


def render_home_page():
    """Render the home page."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## Welcome! 👋")
        st.markdown("""
        This is an intelligent Q&A system built with **Retrieval-Augmented Generation (RAG)** 
        technology to help you explore and understand research papers about:
        
        - 🌍 **Remote Sensing** and satellite imagery
        - 🖼️ **Image Segmentation** techniques
        - 🎯 **Panoptic Segmentation** methods
        - 📊 **Evaluation Metrics** (PQ, mIoU, etc.)
        - 🔄 **Multimodal Fusion** approaches
        - 🤖 **Vision-Language Models** (VLMs)
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. Navigate to **💬 Q&A Interface** in the sidebar
        2. Type your question in the text box
        3. Click **Ask Question** to get an answer
        4. Review the answer and source documents
        5. Ask follow-up questions to dive deeper!
        """)
        
    with col2:
        st.markdown("### 📈 System Status")
        
        settings, vector_manager, rag_graph, success, error = initialize_system()
        
        if success:
            store_info = vector_manager.get_store_info()
            
            st.markdown('<div class="success-box">✅ System Ready</div>', unsafe_allow_html=True)
            
            st.metric("📄 Indexed Documents", f"{store_info['num_vectors']:,}")
            st.metric("🧠 Embedding Model", store_info['embedding_model'])
            st.metric("🤖 LLM Model", settings.llm_model)
        else:
            st.markdown(f'<div class="warning-box">⚠️ {error}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features section
    st.markdown("## ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 Intelligent Search
        Uses state-of-the-art embeddings to find the most relevant 
        information from your research papers.
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Accurate Answers
        Powered by GPT-4o-mini to generate precise, 
        contextual answers based on your documents.
        """)
    
    with col3:
        st.markdown("""
        ### 📚 Source Citations
        Every answer includes references to the source 
        papers and page numbers for verification.
        """)
    
    st.markdown("---")
    
    # Sample questions
    st.markdown("## 💭 Sample Questions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Fundamentals:**
        - What is panoptic segmentation?
        - What's the difference between semantic and instance segmentation?
        - How does multimodal fusion work?
        
        **Methods:**
        - What is Mask2Former?
        - How does Panoptic-DeepLab work?
        - What are transformer-based segmentation models?
        """)
    
    with col2:
        st.markdown("""
        **Datasets & Metrics:**
        - What datasets are available for remote sensing?
        - How is Panoptic Quality (PQ) calculated?
        - What is the OpenEarthMap dataset?
        
        **Advanced:**
        - What is RemoteCLIP?
        - How does GeoRSCLIP improve CLIP?
        - What are vision-language models in remote sensing?
        """)


def render_qa_interface(top_k, temperature):
    """Render the Q&A interface."""
    st.markdown("## 💬 Ask Your Questions")
    
    # Initialize system
    settings, vector_manager, rag_graph, success, error = initialize_system()
    
    if not success:
        st.error(f"❌ System Error: {error}")
        st.info("💡 Please run `python main.py --index` to create the vector database first.")
        return
    
    # Update RAG settings based on sidebar
    rag_graph.top_k = top_k
    rag_graph.temperature = temperature
    
    # Multi-modal input tabs
    st.markdown("### 🤔 Your Question")
    input_tab1, input_tab2 = st.tabs(["📝 Text Only", "🖼️ Text + Image"])
    
    with input_tab1:
        question_text_only = st.text_area(
            "Type your question here:",
            height=100,
            placeholder="Example: What is panoptic segmentation and how is it different from semantic segmentation?",
            help="Ask anything about the research papers in the database",
            key="text_only_question"
        )
    
    with input_tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            question_multimodal = st.text_area(
                "Type your question here:",
                height=100,
                placeholder="Example: Can you explain this diagram from the paper?",
                help="Upload an image and ask questions about it",
                key="multimodal_question"
            )
        with col2:
            uploaded_image = st.file_uploader(
                "Upload an image:",
                type=["png", "jpg", "jpeg", "webp"],
                help="Upload a figure, diagram, or screenshot from papers"
            )
            if uploaded_image:
                # Compatible with both old and new Streamlit versions
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
                st.success("✅ Image uploaded successfully!")
    
    # Determine which question to use based on which tab has content
    # Check if multimodal tab has image uploaded (indicates user is in that tab)
    if uploaded_image:
        question = question_multimodal
    # Otherwise, use whichever question has text content
    elif question_text_only and question_text_only.strip():
        question = question_text_only
        uploaded_image = None
    elif question_multimodal and question_multimodal.strip():
        question = question_multimodal
        uploaded_image = None
    else:
        # No content in either tab
        question = ""
        uploaded_image = None
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        ask_button = st.button("🚀 Ask Question", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear", type="secondary")
    
    if clear_button:
        st.rerun()
    
    # Handle question submission
    if ask_button:
        if not question or not question.strip():
            st.warning("⚠️ Please enter a question first!")
            return
        
        # Show processing status
        spinner_text = "🔍 Searching documents and generating answer..."
        if uploaded_image:
            spinner_text = "🔍 Analyzing image and searching documents..."
        
        with st.spinner(spinner_text):
            start_time = time.time()
            
            try:
                # Query the RAG system (with or without image)
                if uploaded_image:
                    result = rag_graph.query_with_image(question, uploaded_image)
                else:
                    result = rag_graph.query(question)
                
                elapsed_time = time.time() - start_time
                
                # Display answer
                st.markdown("---")
                st.markdown("### 💡 Answer")
                
                if result['errors']:
                    st.error("❌ Errors occurred during processing:")
                    for error in result['errors']:
                        st.error(f"- {error}")
                
                # Display answer with LaTeX support (don't wrap in HTML to allow formula rendering)
                st.success("Answer:")
                st.markdown(result["answer"])
                
                # Display sources
                if result['sources']:
                    st.markdown("### 📚 Source Documents")
                    
                    for idx, source in enumerate(result['sources'], 1):
                        with st.expander(f"📄 Source {idx}: {source['source']} (Page {source['page']})"):
                            st.markdown(f"**Content Preview:**")
                            st.text(source['content_preview'])
                
                # Display metadata
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("⏱️ Response Time", f"{elapsed_time:.2f}s")
                
                with col2:
                    st.metric("📄 Documents Used", len(result['sources']))
                
                with col3:
                    st.metric("📏 Answer Length", f"{len(result['answer'])} chars")
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
    
    # Conversation history (stored in session state)
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if ask_button and question.strip():
        st.session_state.history.append({
            'question': question,
            'answer': result['answer'],
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    # Display history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Question History")
        
        for idx, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"Q{idx}: {item['question'][:60]}... ({item['timestamp']})"):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {item['answer'][:200]}...")


def render_stats_page():
    """Render the system statistics page."""
    st.markdown("## 📊 System Statistics")
    
    settings, vector_manager, rag_graph, success, error = initialize_system()
    
    if not success:
        st.error(f"❌ System Error: {error}")
        return
    
    # Get store info
    store_info = vector_manager.get_store_info()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📄</h3>
            <h2>{store_info['num_vectors']:,}</h2>
            <p>Document Chunks</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢</h3>
            <h2>{store_info['dimension']}</h2>
            <p>Vector Dimension</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯</h3>
            <h2>{settings.top_k_results}</h2>
            <p>Retrieval Top-K</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📏</h3>
            <h2>{settings.chunk_size}</h2>
            <p>Chunk Size</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuration details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🧠 Model Configuration")
        st.markdown(f"""
        - **Embedding Model:** {settings.embedding_model}
        - **LLM Model:** {settings.llm_model}
        - **Temperature:** {settings.temperature}
        - **Chunk Overlap:** {settings.chunk_overlap}
        """)
    
    with col2:
        st.markdown("### 📂 Storage Information")
        vectorstore_path = get_vectorstore_path(settings)
        
        if vectorstore_path.exists():
            index_file = vectorstore_path / "index.faiss"
            pkl_file = vectorstore_path / "index.pkl"
            
            index_size = index_file.stat().st_size / (1024 * 1024) if index_file.exists() else 0
            pkl_size = pkl_file.stat().st_size / (1024 * 1024) if pkl_file.exists() else 0
            
            st.markdown(f"""
            - **Vector Store Path:** {vectorstore_path}
            - **Index Size:** {index_size:.2f} MB
            - **Metadata Size:** {pkl_size:.2f} MB
            - **Total Size:** {index_size + pkl_size:.2f} MB
            """)
    
    st.markdown("---")
    
    # Paper topics
    st.markdown("### 📚 Research Topics Covered")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Segmentation Methods:**
        - Panoptic Segmentation
        - Semantic Segmentation
        - Instance Segmentation
        - Mask2Former
        - Panoptic-DeepLab
        """)
    
    with col2:
        st.markdown("""
        **Datasets:**
        - OpenEarthMap
        - iSAID
        - SEN12MS
        - SpaceNet-8
        """)
    
    with col3:
        st.markdown("""
        **Advanced Topics:**
        - Vision-Language Models
        - RemoteCLIP
        - GeoRSCLIP
        - Multimodal Fusion
        - Transformers
        """)


def render_about_page():
    """Render the about page."""
    st.markdown("## ℹ️ About This System")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is a **Retrieval-Augmented Generation (RAG)** system with **Multimodal Capabilities** 
    designed to help researchers and students explore and understand academic papers about 
    remote sensing and image segmentation.
    
    The system combines:
    - **LangChain** for document processing and chain management
    - **LangGraph** for workflow orchestration
    - **OpenAI GPT-4o** for embeddings, language generation, and **vision analysis**
    - **FAISS** for efficient vector similarity search
    - **Streamlit** for the web interface
    
    ### ✨ New Multimodal Features
    
    - 🖼️ **Image Upload**: Upload diagrams, figures, charts, or screenshots from papers
    - 👁️ **Vision Analysis**: GPT-4o analyzes images and understands visual content
    - 🔄 **Combined Context**: Integrates image analysis with text retrieval from papers
    - 📊 **Visual Understanding**: Explains architectures, metrics, results, and equations in images
    """)
    
    st.markdown("---")
    
    st.markdown("### 🏗️ System Architecture")
    
    st.markdown("""
    **Text-Only Mode:**
    ```
    User Question → Embed Query → Vector Search → Retrieve Docs
                                                        ↓
                                                    Format Context
                                                        ↓
                                                    GPT-4o Generate
                                                        ↓
                                                    Answer + Sources
    ```
    
    **Multimodal Mode (NEW!):**
    ```
    User Question + Image → Encode Image to Base64
                                ↓
                            Vector Search (by text)
                                ↓
                            Retrieve Relevant Docs
                                ↓
                            Combine: Image + Context + Question
                                ↓
                            GPT-4o Vision Analysis
                                ↓
                            Answer (Image + Paper Context) + Sources
    ```
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        - **Python 3.8+**
        - **LangChain** - LLM framework
        - **LangGraph** - Workflow engine
        """)
    
    with col2:
        st.markdown("""
        - **OpenAI API** - Embeddings & LLM
        - **FAISS** - Vector search
        - **Streamlit** - Web interface
        """)
    
    with col3:
        st.markdown("""
        - **PyPDF** - Document processing
        - **Rich** - CLI formatting
        - All docs in **README.md**
        """)
    
    st.markdown("---")
    
    st.markdown("### 👨‍💻 Usage Instructions")
    
    with st.expander("🚀 Quick Start"):
        st.markdown("""
        1. **Index your documents:**
           ```bash
           python main.py --index
           ```
        
        2. **Start the web interface:**
           ```bash
           streamlit run streamlit_app.py
           ```
        
        3. **Open your browser** and navigate to http://localhost:8501
        """)
    
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        **Text-Only Queries:**
        - Ask **specific questions** rather than broad ones
        - Include **key terms** from your research area
        - Try **follow-up questions** to dive deeper
        - Use the **top-k slider** to adjust context
        - Lower **temperature** for more factual answers
        
        **Multimodal Queries (Text + Image):**
        - Upload clear, high-quality images (diagrams, charts, figures)
        - Ask specific questions about the image content
        - Combine image analysis with paper context queries
        - Use for: architecture diagrams, result tables, metric charts, equations
        - Example: "Explain this architecture and compare it to methods in the papers"
        """)
    
    with st.expander("❓ Troubleshooting"):
        st.markdown("""
        **Vector store not found:**
        - Run `python main.py --index` first
        
        **Slow responses:**
        - Reduce top-k value
        - Check internet connection
        - Verify OpenAI API key
        
        **Poor answer quality:**
        - Increase top-k for more context
        - Try rephrasing your question
        - Check if topic is covered in papers
        """)
    
    st.markdown("---")
    
    st.markdown("### 📜 License & Credits")
    
    st.markdown("""
    This project is licensed under the **MIT License**.
    
    **Powered by:**
    - OpenAI GPT Models
    - LangChain Framework
    - Streamlit Framework
    - FAISS by Facebook Research
    
    **Built with** ❤️ **for researchers and students**
    """)


def main():
    """Main application entry point."""
    render_header()
    
    # Render sidebar and get settings
    page, top_k, temperature = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        render_home_page()
    elif page == "💬 Q&A Interface":
        render_qa_interface(top_k, temperature)
    elif page == "📊 System Stats":
        render_stats_page()
    elif page == "ℹ️ About":
        render_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; padding: 1rem;">'
        '🛰️ Remote Sensing Papers Q&A System | '
        'Powered by RAG, LangChain & LangGraph | '
        f'© {datetime.now().year}'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

