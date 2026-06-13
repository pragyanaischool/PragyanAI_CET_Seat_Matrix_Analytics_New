import streamlit as st
import pandas as pd
import sqlite3
from engines.enrichment_engine import CollegeEnrichmentEngine
from database.db_handler import save_enrichment_record, DB_NAME

def render_intelligence_center():
    """
    Renders the Federated Knowledge Discovery and Accreditation 
    Harvesting interface panel.
    """
    st.set_page_config(
        page_title="PragyanAI College Intelligence",
        page_icon="🏛️",
        layout="wide"
    )

    # 1. Page Header Branding
    st.title("🏛️ Knowledge Graph Discovery & Enrichment Center")
    st.subheader("Verify institutional metrics and pull rankings concurrently from Google, DuckDuckGo, and Wikipedia.")
    st.markdown("---")

    # 2. Extract uniquely registered campuses from the active data lake
    try:
        conn = sqlite3.connect(DB_NAME)
        # Select distinct pairs to prevent redundant API queries for different branches
        colleges_df = pd.read_sql_query(
            "SELECT DISTINCT college_name, city FROM seat_matrix ORDER BY college_name ASC", 
            conn
        )
        conn.close()
    except Exception as db_err:
        st.error(f"Database Connectivity Failure: Could not extract institutional lists: {str(db_err)}")
        colleges_df = pd.DataFrame()

    # 3. Workspace Flow Validation Splitter
    if colleges_df.empty:
        st.info("💡 Central database lake is currently empty. Please upload and parse seat matrix files inside Page 1 first.")
        return

    st.markdown("### 🔍 Select and Enrich Institutional Profiles")
    st.caption("⚡ Powered concurrently by Google Search API, DuckDuckGo Search, and Wikipedia API Wrappers")

    # Create a user-friendly selection drop-down combining name and municipality
    college_options = colleges_df.apply(lambda r: f"{r['college_name']} ({r['city']})", axis=1).tolist()
    selected_option = st.selectbox(
        "Choose target campus profile entity to query open-web knowledge charts:",
        options=college_options,
        help="Select the exact college entry you want to enrich with NAAC, NBA, and NIRF metrics."
    )

    # Extract the true unmapped query strings based on index matching positions
    selected_idx = college_options.index(selected_option)
    target_college_name = colleges_df.iloc[selected_idx]['college_name']
    target_college_city = colleges_df.iloc[selected_idx]['city']

    # Layout design partition showing current local data versus query operations
    col_layout_left, col_layout_right = st.columns([2, 3])

    with col_layout_left:
        st.markdown("#### 🏛️ Selected Profile Context")
        st.info(f"""
        **Official Parsed Name:** `{target_college_name}`  
        
        **Campus Municipality/City:** `{target_college_city}`
        """)
        
        # Trigger button for federated searching
        execute_search_trigger = st.button(
            f"🌐 Run Federated Web Search Discovery Engine", 
            type="primary", 
            use_container_width=True
        )

    with col_layout_right:
        st.markdown("#### 💎 Live Enrichment Results Output")
        
        if execute_search_trigger:
            with st.spinner(f"Scouring web nodes, index records, and encyclopedias for '{target_college_name}'..."):
                try:
                    # Initialize our multi-engine search coordinator class instance
                    enrichment_processor = CollegeEnrichmentEngine()
                    
                    # Execute synchronous search & extraction routine loops
                    enriched_result_payload = enrichment_processor.discover_college_details(
                        target_college_name, 
                        target_college_city
                    )
                    
                    # Step 4: Commit parsed parameters right into the cache database
                    save_enrichment_record(enriched_result_payload)
                    
                    st.success("🎉 Target institution metrics successfully harvested and synchronized with local cache repositories!")
                    
                    # Display structured output fields elegantly via metrics cards
                    st.json(enriched_result_payload)
                    
                except Exception as enrichment_err:
                    st.error(f"Enrichment Execution Intercepted Exception: {str(enrichment_err)}")
        else:
            st.write("Click the button on the left to pull real-time accreditation data and website URLs.")

    st.markdown("---")

    # 5. Master Enriched Ledger Cache View
    st.markdown("### 📑 Master Enriched Intelligence Cache Registry")
    st.markdown("Review all institutional profiles currently enriched and verified across the data lake:")
    
    try:
        conn = sqlite3.connect(DB_NAME)
        master_registry_df = pd.read_sql_query("SELECT * FROM college_enrichment ORDER BY college_name ASC", conn)
        conn.close()
        
        if not master_registry_df.empty:
            # Format columns beautifully to emphasize analytics parameters
            st.dataframe(
                master_registry_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "college_name": st.column_config.TextColumn("Verified Institution Name", width="medium"),
                    "website": st.column_config.LinkColumn("Official Portal domain", width="small"),
                    "naac_rating": st.column_config.TextColumn("NAAC Grade", width="small"),
                    "nba_accredited": st.column_config.TextColumn("NBA Accreditation", width="small"),
                    "nirf_ranking": st.column_config.TextColumn("NIRF Ranking Index", width="small"),
                    "summary": st.column_config.TextColumn("Executive Summary Profile", width="large")
                }
            )
        else:
            st.warning("No enriched cache ledger records exist inside the local storage engine yet. Run an open-web query above.")
    except Exception as registry_err:
        st.error(f"Failed to load systemic metadata cache tables: {str(registry_err)}")

if __name__ == "__main__":
    render_intelligence_center()
  
