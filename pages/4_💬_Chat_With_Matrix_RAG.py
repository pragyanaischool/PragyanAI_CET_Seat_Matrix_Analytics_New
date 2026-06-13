import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from database.db_handler import get_combined_analytics

def render_rag_chat_workspace():
    """
    Renders the Conversational semantic RAG workspace page panel,
    leveraging local Hugging Face embedding pipelines and a FAISS Vector Store.
    """
    st.set_page_config(
        page_title="PragyanAI Semantic RAG Chat",
        page_icon="💬",
        layout="wide"
    )

    # 1. Page Header Branding UI Block
    st.title("💬 Conversational Semantic RAG Workspace")
    st.subheader("Query multi-year seat counts, track branch availability changes, and extract trends using semantic search.")
    st.markdown("---")

    # Fetch combined dataset (Seat Matrix records + open-web scraped rankings)
    df_lake = get_combined_analytics()

    if df_lake.empty:
        st.info("💡 The semantic conversational system requires active database layers. Complete initial file parsing under Page 1 first.")
        return

    # --- PHASE 1: CACHED NEURAL VECTOR INDEX LIFECYCLE MANAGEMENT ---

    @st.cache_resource(show_spinner="📥 Initializing Hugging Face MiniLM Transformer Embedding Model...")
    def get_hf_embedding_pipeline():
        """
        Loads the all-MiniLM-L6-v2 sentence transformer directly from Hugging Face Hub.
        Computes 384-dimensional mathematical vector arrays locally on host CPU lines.
        """
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

    @st.cache_data(show_spinner="⚡ Compiling data objects and seeding local FAISS Vector Storage index...")
    def build_vector_search_index(_df):
        """
        Serializes relational database table rows into distinct document blocks
        and embeds them securely into an in-memory Facebook AI Similarity Search (FAISS) store.
        """
        document_chunks = []
        
        for _, row in _df.iterrows():
            # Standardize column structures into descriptive paragraphs for semantic matching
            descriptive_payload = f"""
            Intake Academic Year: {row.get('intake_year', 'N/A')}
            College Name: {row.get('college_name', 'N/A')}
            Municipality City: {row.get('city', 'N/A')}
            Geographic District: {row.get('district', 'N/A')}
            Physical Postal Address: {row.get('address', 'N/A')}
            Engineering Branch/Department Name: {row.get('dept', 'N/A')}
            Total Admitted Seat Capacity: {row.get('intake', 0)} Seats
            Official Portal Link Website URL: {row.get('website', 'N/A')}
            Institutional NAAC Rating Grade: {row.get('naac_rating', 'N/A')}
            NBA Accreditation Parameters: {row.get('nba_accredited', 'N/A')}
            NIRF National Ranking Standings Index: {row.get('nirf_ranking', 'N/A')}
            Campus Executive Summary Profile: {row.get('summary', 'N/A')}
            """.strip()
            
            # Map structural identification dictionaries down into the retrieval item metadata layer
            metadata_link = {
                "college": str(row.get('college_name', '')),
                "year": str(row.get('intake_year', ''))
            }
            document_chunks.append(Document(page_content=descriptive_payload, metadata=metadata_link))
            
        embedding_model = get_hf_embedding_pipeline()
        # Vectorize text documents and instantiate the FAISS index structure
        faiss_store = FAISS.from_documents(document_chunks, embedding_model)
        return faiss_store

    # Bootstrap the FAISS index vector database using our cached data handler
    try:
        vector_db_index = build_vector_search_index(df_lake)
    except Exception as index_err:
        st.error(f"Failed to compile mathematical vector database index maps: {str(index_err)}")
        st.stop()

    # --- PHASE 2: MULTI-TURN CHAT INTERACTIVE PERSISTENCE SPACE ---

    # Instantiate or pull the multi-turn session chat arrays
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history = []

    # Display running contextual conversation threads back to the UI panel workspace
    for message in st.session_state.rag_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Continuous chat prompt input trigger check loop
    if user_prompt := st.chat_input("Ask a question (e.g., 'Compare the total seats for Computer Science between 2024 and 2026')"):
        # Instantly echo user input characters to the web chat display
        with st.chat_message("user"):
            st.markdown(user_prompt)
        st.session_state.rag_chat_history.append({"role": "user", "content": user_prompt})

        # --- PHASE 3: SEMANTIC RETRIEVAL CONTEXT EXTRACTION LAYER ---
        with st.chat_message("assistant"):
            response_container = st.empty()
            
            with st.spinner("Executing k-nearest neighbor similarity lookup across FAISS index..."):
                # Execute Cosine/Euclidean proximity searches to pull the top 6 closest matching row blocks
                relevant_documents = vector_db_index.similarity_search(user_prompt, k=6)
                
                # Consolidate retrieved documents into a single prompt block
                retrieved_context_block = "\n\n=== RETRIEVED ROW PARAMETERS ENTRY ===\n".join(
                    [doc.page_content for doc in relevant_documents]
                )

            # --- PHASE 4: GENERATIVE LLM ADMISSIONS INSIGHT CHIPS ---
            with st.spinner("Synthesizing multi-year analysis report lines via Llama3..."):
                
                system_prompt_instruction = f"""
                You are PragyanAI's senior admissions data analyst and system strategist. 
                Your task is to accurately fulfill user prompts by thoroughly evaluating the parsed multi-year seat matrix records and open-web accreditation variables retrieved from our local FAISS index framework below.
                
                --- RETRIEVED SEMANTIC CONTEXT (SQLite Data Lake Slices) ---
                {retrieved_context_block}
                -----------------------------------------------------------
                
                Guidelines for response construction:
                1. Provide specific institution names, year parameters, and exact numerical intake counts whenever present in the context.
                2. If the prompt implies analytical data joins across multiple years (such as tracking changes, additions, or capacity drop-offs from 2024 to 2026), perform calculations and format your answer as a clear table.
                3. Base your logic directly on facts found within the retrieval block. If details are unmentioned, rely on general knowledge but explicitly disclaim it.
                4. Keep responses highly professional, clean, and organized using markdown lists or structured layouts.
                """
                
                try:
                    # Initialize high-capacity Llama3-70B engine to enforce reasoning limits
                    inference_engine = ChatGroq(model_name="llama3-70b-8192", temperature=0.1)
                    
                    message_payload = [
                        {"role": "system", "content": system_prompt_instruction},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    # Generate completion
                    ai_response_content = inference_engine.invoke(message_payload).content
                    response_container.markdown(ai_response_content)
                    
                    # Save the assistant response back into the persistent session thread
                    st.session_state.rag_chat_history.append({"role": "assistant", "content": ai_response_content})
                    
                except Exception as llm_err:
                    response_container.empty()
                    st.error(f"Inference Engine Failed to generate response frames: {str(llm_err)}")

if __name__ == "__main__":
    render_rag_chat_workspace()
  
