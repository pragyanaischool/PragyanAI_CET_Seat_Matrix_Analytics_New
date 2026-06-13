import streamlit as st
import os
import sys

# ==============================================================================
# 🎯 PATH PATCH & CACHE PROTECTION LAYER (PREVENTS INTERMITTENT KEYERRORS)
# ==============================================================================
# Resolve absolute path bounds to guarantee module accessibility
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Flush out lingering cached runtime instances to eliminate cross-page import collisions
for module_key in list(sys.modules.keys()):
    if (
        module_key.startswith("engines") 
        or module_key.startswith("database") 
        or "matrix_parser" in module_key
        or "ensemble_orchestrator" in module_key
        or "enrichment_engine" in module_key
    ):
        sys.modules.pop(module_key, None)
# ==============================================================================

import pandas as pd
from engines.ensemble_orchestrator import PragyanEnsembleParser
from engines.enrichment_engine import PragyanMatrixEnricher
from database.db_handler import save_matrix_records

def render_ingestion_portal():
    """
    Main Ingestion View Portal for PragyanAI.
    Combines multi-engine layout extraction layers and entity enrichment 
    to guarantee clean row multiplication metrics without signature drops.
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

            # 2. Pipeline Segment Beta: Cross-Validation & Text Enrichment Rails
            with st.spinner("🧠 Running data healing rules (Standardizing geographics, abbreviations, and intake sanity filters)..."):
                try:
                    enricher = PragyanMatrixEnricher()
                    clean_normalized_df = enricher.enrich_extracted_dataframe(raw_extracted_df)
                except Exception as enrichment_err:
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
                st.warning("💡 Advice: Ensure the source document matches KEA target signatures and that valid AI API credits are assigned inside your .env configuration.")
    else:
        st.info("💡 Standby: Upload a source PDF seat matrix document file layout above to open execution controls.")

if __name__ == "__main__":
    render_ingestion_portal()
