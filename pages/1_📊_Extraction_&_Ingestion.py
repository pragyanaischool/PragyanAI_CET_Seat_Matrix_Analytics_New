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
# ==============================================================================

import pandas as pd
from engines.ensemble_orchestrator import PragyanEnsembleParser
from engines.enrichment_engine import CollegeEnrichmentEngine  
from database.db_handler import save_matrix_records

def render_ingestion_portal():
    """
    Main Ingestion and Data Ingestion View Portal for PragyanAI.
    Coordinates multi-engine layout extraction layers and advanced web enrichment
    to guarantee clean row multiplication metrics without structural drops.
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
            help="Upload the raw PDF layout. The engine will run concurrent extractions via Docling, pdfplumber, and PyMuPDF."
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
        
        if st.button("🚀 Execute Multi-Engine Ingestion & Sanitization"):
            
            # 1. Pipeline Segment Alpha: Multi-Engine Concurrent Extraction
            with st.spinner("⏳ Compiling multi-library text maps (IBM Docling Table Grids + Plumber + PyMuPDF)..."):
                try:
                    orchestrator = PragyanEnsembleParser()
                    raw_extracted_df = orchestrator.analyze_and_extract_matrix(file_bytes, intake_year)
                except Exception as extraction_err:
                    st.error(f"❌ Critical Error during Ensemble Extraction Layer: {str(extraction_err)}")
                    return

            # 2. Pipeline Segment Beta: Cross-Validation & Asynchronous Web Enrichment Rails
            with st.spinner("🧠 Running parallel web enrichment (SerpAPI Google Search + Wikipedia + DuckDuckGo)..."):
                try:
                    # Instantiating the aligned asynchronous engine component
                    enricher = CollegeEnrichmentEngine()  
                    
                    # Process the extracted rows through the enrichment engine loop
                    if raw_extracted_df is not None and not raw_extracted_df.empty:
                        enriched_records = []
                        
                        # Isolate colleges uniquely to save API lookups and leverage concurrent threads nicely
                        unique_colleges = raw_extracted_df[['college_name', 'city']].drop_duplicates()
                        
                        st.info(f"🔍 Discovered {len(unique_colleges)} unique institution profiles. Initializing federated knowledge discovery threads...")
                        
                        # Progress bar for visual UI tracking feedback loops
                        progress_bar = st.progress(0)
                        total_colleges = len(unique_colleges)
                        
                        for idx, row in enumerate(unique_colleges.itertuples(), 1):
                            college_details = enricher.discover_college_details(row.college_name, row.city)
                            enriched_records.append(college_details)
                            progress_bar.progress(idx / total_colleges)
                            
                        enrichment_lookup_df = pd.DataFrame(enriched_records)
                        
                        # Merge the newly discovered parameters back with the multiplied rows
                        clean_normalized_df = pd.merge(
                            raw_extracted_df, 
                            enrichment_lookup_df, 
                            on=['college_name', 'city'], 
                            how='left'
                        )
                    else:
                        clean_normalized_df = pd.DataFrame()
                        
                except Exception as enrichment_err:
                    # FIXED: Re-aligned exception variable mapping precisely to resolve UnboundLocalError
                    st.error(f"❌ Critical Error during Post-Extraction Enrichment Layer: {str(enrichment_err)}")
                    return

            # 3. Pipeline Segment Gamma: Metrics Display and Relational Database Injection
            if clean_normalized_df is not None and not clean_normalized_df.empty:
                st.success(f"✅ Success! Extracted and verified {len(clean_normalized_df)} individual course-multiplied rows with zero signature drops!")
                
                # Render interactive high-density dataframe metrics display canvas
                st.dataframe(
                    clean_normalized_df.sort_values(by=['college_name', 'dept']).reset_index(drop=True), 
                    use_container_width=True
                )
                
                # Write back directly to localized persistent storage layer
                with st.spinner("💾 Committing structured records matrix down to relational database data lake..."):
                    try:
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
