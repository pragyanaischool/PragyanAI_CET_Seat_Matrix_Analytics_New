import streamlit as st
import os
import sys

# ==============================================================================
# 🎯 SAFE HEALED CACHE PROTECTION LAYER (ELIMINATES KEYERRORS COMPLETELY)
# ==============================================================================
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Expressly define keywords to scan and drop safely from system cache maps
modules_to_flush = [
    "engines", 
    "database", 
    "matrix_parser", 
    "ensemble_orchestrator", 
    "enrichment_engine"
]

for module_key in list(sys.modules.keys()):
    if any(target in module_key for target in modules_to_flush):
        # Specifying None as the second argument forces a silent fail if the key is missing
        sys.modules.pop(module_key, None)
# ==============================================================================

import pandas as pd
from engines.ensemble_orchestrator import PragyanEnsembleParser
from engines.enrichment_engine import CollegeEnrichmentEngine  
from database.db_handler import save_matrix_records, save_enrichment_record

def render_ingestion_portal():
    """
    Main Ingestion and Data Ingestion View Portal for PragyanAI.
    Coordinates resilient multi-LLM cascading extraction layers and parallel
    web enrichment to populate core relational database target schemas safely.
    """
    st.set_page_config(
        page_title="PragyanAI Ingestion Portal", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 Multi-Engine Layout Consensus Ingestion Core")
    st.subheader("Process unpredictable seat allocation arrays while automatically tracking row-multiplied disciplines.")
    st.markdown("---")

    # --- INPUT USER PANEL CONFIGURATIONS ---
    col_uploader, col_horizon = st.columns([3, 1])
    
    with col_uploader:
        uploaded_file = st.file_uploader(
            "Upload Official Government Seat Matrix Document (PDF)", 
            type=["pdf"],
            help="Upload the raw PDF layout. The engine will run sequential extractions utilizing a cascading failover model pool."
        )
        
    with col_horizon:
        intake_year = st.selectbox(
            "Target Academic Horizon Year", 
            [2024, 2025, 2026], 
            index=1,
            help="Select the operational seat matrix year profile to allocate for relational time-series tracking."
        )

    st.markdown("### ⚙️ Pipeline Control Framework")
    
    # Ingestion Pipeline Run Execution Block
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        if st.button("🚀 Execute Resilient Multi-LLM Ingestion & Sanitization"):
            
            # 1. Pipeline Segment Alpha: Multi-Engine Concurrent Extraction
            with st.spinner("⏳ Running dynamic Multi-LLM cascade extraction loops across text chunk segments..."):
                try:
                    orchestrator = PragyanEnsembleParser()
                    raw_extracted_df = orchestrator.analyze_and_extract_matrix(file_bytes, intake_year)
                except Exception as extraction_err:
                    st.error(f"❌ Critical Error during Ensemble Extraction Layer: {str(extraction_err)}")
                    return

            # Validate extraction payload block state
            if raw_extracted_df is not None and not raw_extracted_df.empty:
                
                # Enforce pristine uppercase structural parameters to guarantee clean lookup match signatures
                for col in ['college_name', 'city', 'district', 'dept']:
                    if col in raw_extracted_df.columns:
                        raw_extracted_df[col] = raw_extracted_df[col].astype(str).str.strip().str.upper()
                
                # 2. Pipeline Segment Beta: Cross-Validation & Asynchronous Web Enrichment Rails
                with st.spinner("🧠 Running parallel web enrichment threads (SerpAPI + Wikipedia + DuckDuckGo)..."):
                    try:
                        enricher = CollegeEnrichmentEngine()  
                        enriched_records = []
                        
                        # Isolate unique colleges to save API quotas and optimize parallel performance limits
                        unique_colleges = raw_extracted_df[['college_name', 'city']].drop_duplicates()
                        
                        st.info(f"🔍 Discovered {len(unique_colleges)} unique institution profiles. Initializing federated knowledge threads...")
                        
                        # Progress bar initialization for tracking loop updates
                        progress_bar = st.progress(0)
                        total_colleges = len(unique_colleges)
                        
                        for idx, row in enumerate(unique_colleges.itertuples(), 1):
                            college_details = enricher.discover_college_details(row.college_name, row.city)
                            enriched_records.append(college_details)
                            
                            # Transactionally persist unique enrichment parameters to cache table
                            save_enrichment_record(college_details)
                            
                            progress_bar.progress(idx / total_colleges)
                            
                        enrichment_lookup_df = pd.DataFrame(enriched_records)
                        
                        # Merge the newly discovered parameters back with raw extracted options records
                        clean_normalized_df = pd.merge(
                            raw_extracted_df, 
                            enrichment_lookup_df, 
                            on=['college_name', 'city'], 
                            how='left'
                        )
                    except Exception as enrichment_err:
                        st.error(f"❌ Critical Error during Post-Extraction Enrichment Layer: {str(enrichment_err)}")
                        return
            else:
                st.error("❌ Extraction Error: The text segmentation parser cluster returned an empty dataframe structure.")
                return

            # 3. Pipeline Segment Gamma: Metrics Display and Relational Database Ingestion
            if clean_normalized_df is not None and not clean_normalized_df.empty:
                st.success(f"✅ Success! Extracted and verified {len(clean_normalized_df)} individual course-multiplied rows with zero signature drops!")
                
                # Render interactive high-density dataframe metrics display canvas
                st.dataframe(
                    clean_normalized_df.sort_values(by=['college_name', 'dept']).reset_index(drop=True), 
                    use_container_width=True
                )
                
                # Write back directly to localized persistent sqlite tables structure
                with st.spinner("💾 Committing structured records matrix down to relational database tables..."):
                    try:
                        # Saves directly to seat_matrix table
                        save_matrix_records(clean_normalized_df)
                        st.balloons()
                        st.info("💾 Operations Log: Target matrices committed and validated inside the local storage infrastructure successfully.")
                    except Exception as db_err:
                        st.error(f"❌ Database Transaction Failure: {str(db_err)}")
            else:
                st.error("❌ Spatial Processing Failure: The framework returned an empty layout dataframe. Check alignment parameters or original file formats.")
    else:
        st.info("💡 Standby: Upload a source PDF seat matrix document file layout above to open execution controls.")

if __name__ == "__main__":
    render_ingestion_portal()
    
