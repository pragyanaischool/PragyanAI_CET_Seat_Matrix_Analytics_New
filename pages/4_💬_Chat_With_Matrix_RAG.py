import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from database.db_handler import get_combined_analytics

def render_rag_chat_workspace():
    """
    Renders the Conversational Neural Semantic RAG Workspace panel,
    leveraging local Hugging Face embedding pipelines and an in-memory FAISS store.
    """
    st.set_page_config(
        page_title="Neural RAG Chatbot",
        page_icon="💬",
        layout="wide"
    )

    # 1. Page Header Branding UI Block
    st.title("💬 Neural Semantic RAG Conversational Workspace")
    st.subheader("Query multi-year seat parameters, track discipline availability trends, and audit rankings using natural language.")
    st.markdown("---")

    # Fetch combined dataset (Relational Seat Matrices + Open-Web Scraped Accreditations)
    df_lake = get_combined_analytics()

    # 2. Flow Control Boundary Check
    if df_lake.empty:
        st.info("💡 The conversational semantic engine requires data records inside SQLite. Parse seat matrix documents under Page 1 first.")
        return

    # --- PHASE 1: CACHED NEURAL VECTOR INDEX LIFECYCLE MANAGEMENT ---

    @st.cache_resource(show_spinner="📥 Initializing Hugging Face MiniLM Dense Transformer Embedding Model...")
    def get_hf_embedding_pipeline():
        """
        Loads the sentence-transformer weights directly from Hugging Face Hub.
        Generates dense vector layers locally on host CPU frames to eliminate third-party API costs.
        """
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    @st.cache_data(show_spinner="⚡ Generating mathematical vector fields and seeding local FAISS database index...")
    def build_vector_search_index(_df):
        """
        Serializes multi-column relational row fields into cohesive descriptive text chunks
        and indexes them into a high-performance FAISS instance.
        """
        document_chunks = []
        
        for _, row in _df.iterrows():
            # Standardize disjointed dataframe rows into clean plain-text descriptive strings
            descriptive_payload = f"""
            Intake Academic Year: {row.get('intake_year', 'N/A')}
            College Name: {row.get('college_name', 'N/A')}
            Municipality City: {row.get('city', 'N/A')}
            Geographic District: {row.get('district', 'N/A')}
            Physical Address Location: {row.get('address', 'N/A')}
            Engineering Branch Department: {row.get('dept', 'N/A')}
            Total Intake Capacity Allocation: {row.get('intake', 0)} Seats
            Official Domain Website Link: {row.get('website', 'N/A')}
            Institutional NAAC Rating Grade: {row.get('naac_rating', 'N/A')}
            NBA Accreditation Parameters: {row.get('nba_accredited', 'N/A')}
            NIRF National Ranking Standing Index: {row.get('nirf_ranking', 'N/A')}
            Campus Summary Description Profile: {row.get('summary', 'N/A')}
            """.strip()
            
            # Pack key identifiers into metadata fields for targeted tracking structures
            metadata_link = {
                "college": str(row.get('college_name', '')),
                "year": str(row.get('intake_year', ''))
            }
            document_chunks.append(Document(page_content=descriptive_payload, metadata=metadata_link))
            
        embedding_model = get_hf_embedding_pipeline()
        # Vectorize document structures and instantiate the FAISS index
        faiss_store = FAISS.from_documents(document_chunks, embedding_model)
        return faiss_store

    # Instantiate or pull the cached reference map state of our vector database index
    try:
        vector_db_index = build_vector_search_index(df_lake)
    except Exception as index_err:
        st.error(f"Failed to compile operational matrix vector maps: {str(index_err)}")
        st.stop()

    # --- PHASE 2: MULTI-TURN CHAT INTERACTIVE PERSISTENCE SPACE ---

    # Initialize session history tracking state arrays if absent from the application scope
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # Print rolling conversation dialogue strings down into the active viewport layout
    for message in st.session_state.rag_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Streamlit reactive input element check loop hook
    if user_prompt := st.chat_input("Query anything (e.g., 'Summary of seat modifications for COMPUTER SCIENCE from 2024 to 2026')"):
        # Instantly echo user input characters to the web interface display
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.rag_chat_history.append({"role": "user", "content": user_prompt})

        # --- PHASE 3: SEMANTIC RETRIEVAL CONTEXT EXTRACTION LAYER ---
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            with st.spinner("Extracting k-nearest neighbor similarity contexts from FAISS vector spaces..."):
                # Run proximity distance operations to retrieve the top 6 closest matching text data slices
                relevant_documents = vector_db_index.similarity_search(user_prompt, k=6)
                
                # Consolidate retrieved pieces into a single unified context prompt block
                retrieved_context_block = "\n\n=== RETRIEVED ROW PARAMETERS ENTRY ===\n".join(
                    [doc.page_content for doc in relevant_documents]
                )

            # --- PHASE 4: GENERATIVE LLM ADMISSIONS INSIGHT CHIPS ---
            with st.spinner("Synthesizing multi-year analysis report lines via Llama3..."):
                
                system_prompt_instruction = f"""
                You are PragyanAI's senior admissions data intelligence consultant assistant. 
                Your task is to accurately fulfill user prompts by thoroughly evaluating the parsed multi-year seat matrix records and open-web accreditation variables retrieved from our local FAISS index framework below.
                
                --- RETRIEVED SEMANTIC CONTEXT (SQLite Data Lake Slices) ---
                {retrieved_context_block}
                -----------------------------------------------------------
                
                Guidelines for response construction:
                1. Provide specific institution names, year parameters, and exact numerical intake counts whenever present in the context.
                2. If the prompt implies analytical data joins across multiple years (such as tracking changes, additions, or capacity drop-offs from 2024 to 2026), perform calculations and format your answer as a clear markdown table.
                3. Base your logic directly on facts found within the retrieval block. If details are unmentioned inside the text, rely on general knowledge but explicitly disclaim it.
                4. Keep responses highly professional, clean, and organized using markdown lists or structured layouts.
                """
                
                try:
                    # Initialize high-capacity Llama3-70B engine to enforce strict reasoning limits
                    inference_engine = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)
                    
                    message_payload = [
                        {"role": "system", "content": system_prompt_instruction},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Generate completion stream responses
                    ai_response_content = inference_engine.invoke(message_payload).content
                    response_container.markdown(ai_response_content)
                    
                    # Save the assistant response back into the persistent session thread
                    st.session_state.rag_chat_history.append({"role": "assistant", "content": ai_response_content})
                    
                except Exception as llm_err:
                    response_container.empty()
                    st.error(f"Inference Engine Connection Interrupted: {str(llm_err)}")

if __name__ == "__main__":
    render_rag_chat_workspace()
    
