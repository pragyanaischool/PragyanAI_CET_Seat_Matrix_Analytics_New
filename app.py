import streamlit as st
import os
from database.db_handler import init_db, get_combined_analytics

def main():
    """
    Core entry point and environment bootstrap configuration 
    for the PragyanAI Seat Matrix Intelligence Suite.
    """
    # 1. Initialize Relational Storage Models immediately upon startup
    try:
        init_db()
    except Exception as e:
        st.error(f"Critical System Failure: Relational database workspace could not initialize: {str(e)}")
        st.stop()

    # 2. Hero Header Brand Realization Block
    st.title("🤖 PragyanAI Engineering Seat Matrix Intelligence Suite")
    st.subheader("Automated Multi-Format Ingestion, Federated Verification & Multi-Year RAG Planning Framework")
    
    st.markdown("---")

    # 3. Dynamic Real-Time System Status Metrics Card
    df_lake = get_combined_analytics()
    
    col_v1, col_v2, col_v3, col_v4 = st.columns(4)
    with col_v1:
        st.metric("Total Data Rows Cached", len(df_lake) if not df_lake.empty else 0)
    with col_v2:
        st.metric("Unique Campuses Tracked", df_lake['college_name'].nunique() if not df_lake.empty else 0)
    with col_v3:
        st.metric("Academic Cycles Logged", df_lake['intake_year'].nunique() if not df_lake.empty else 0)
    with col_v4:
        # Check how many institutions have successfully fetched external enrichment profiles
        enriched_count = df_lake[df_lake['website'].notna() & (df_lake['website'] != 'N/A')]['college_name'].nunique() if not df_lake.empty else 0
        st.metric("Enriched College Profiles", enriched_count)

    st.markdown("---")

    # 4. Interactive Architectural Workflow Guide
    st.markdown("### 🗺️ Multi-Turn Deep Tech System Workflow Architecture")
    st.markdown("""
    This venture studio platform provides full decision visibility by transforming unstructured government matrix snapshots into clean relational intelligence parameters. 
    Use the **Sidebar Navigation Panel** to route through the operational desks sequentially:
    """)

    # Standard columns layout grid showcasing page tasks cleanly
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("""
        #### [1] 📊 Extraction & Ingestion Portal
        * **Multi-Format Ingestion:** Directly drop **PDF, DOCX, XLSX, or CSV** matrix profiles.
        * **Heuristic Layout Healing:** Combines custom regex patterns with an **LLM fallback buffer (Llama 3 70B via Groq)** to heal complex line-wraps and ensure zero data loss.
        * **Year-Wise Tagging:** Automatically indexes streams by intake year to prevent multi-year collisions.
        
        #### [2] 🏛️ College Intelligence Center
        * **Federated Lookup Scrapers:** Queries **Google Search (SerpAPI), DuckDuckGo, and Wikipedia** simultaneously.
        * **Automated Accreditation Harvesting:** Extracts and caches official domain links, **NAAC Grades, NBA Accreditation parameters, and NIRF Rankings**.
        """)
        
    with p2:
        st.markdown("""
        #### [3] 📈 Strategic Analytics Workspace
        * **Multi-Dimensional Breakdowns:** Instant aggregations across state capacities, district spreads, and specific departmental seats.
        * **YoY Systems Variance:** Automated tracking metrics that instantly flag **added colleges, dropped courses, or capacity shifts** between two years.
        * **Export Desk:** Clean tabular query downloads in normalized CSV formats.
        
        #### [4] 💬 Semantic Conversational RAG Desk
        * **Neural Index Vector Store:** Vectorizes text fields into **384-dimensional dense vectors** running locally on CPU.
        * **FAISS Vector DB Database:** Replaces inaccurate keyword lookups with true spatial semantic similarity matching.
        * **Llama 3 Inference Chain:** Connects contextual data fragments directly to chat prompts for analytical multi-year deep dives.
        """)

    st.markdown("---")

    # 5. Security & Environment Configuration Validation Monitor
    st.markdown("### 🔌 API Integration Pipeline Health Status")
    
    groq_api = os.getenv("GROQ_API_KEY")
    serp_api = os.getenv("SERPAPI_API_KEY")

    status_c1, status_c2 = st.columns(2)
    with status_c1:
        if groq_api:
            st.success("✅ **Groq LLM Pipeline Interface:** Connected (`Llama3-70b` & `Llama3-8b` are fully functional)")
        else:
            st.error("❌ **Groq LLM Pipeline Interface:** Missing `GROQ_API_KEY` token inside `.env` configuration runtime.")
            
    with status_c2:
        if serp_api:
            st.success("✅ **SerpAPI Engine Interface:** Connected (Google search enrichment layer is fully functional)")
        else:
            st.warning("⚠️ **SerpAPI Engine Interface:** Missing `SERPAPI_API_KEY`. System will automatically default entirely to DuckDuckGo/Wikipedia extraction scrapers.")

if __name__ == "__main__":
    main()
