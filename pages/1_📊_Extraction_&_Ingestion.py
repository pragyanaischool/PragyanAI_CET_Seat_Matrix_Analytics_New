import streamlit as st
import os
import sys

# ==============================================================================
# 🎯 PATH PATCH & CACHE PROTECTION LAYER (FIX FOR IMPORT KEYERRORS)
# ==============================================================================
# 1. Deduce the absolute root directory footprint paths
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 2. Inject parent root framework path into global execution lookup directories
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# 3. FIX: Completely drop any cached or partial module instances to prevent KeyErrors
for module_key in list(sys.modules.keys()):
    if (
        module_key.startswith("engines") 
        or module_key.startswith("database") 
        or "matrix_parser" in module_key
    ):
        sys.modules.pop(module_key, None)
# ==============================================================================

# Now completely safe to proceed with standard package references
import pandas as pd
from engines.document_extractor import PragyanDocumentExtractor
from engines.matrix_parser import CETSeatMatrixParser
from database.db_handler import save_matrix_records

def render_ingestion_portal():
    st.set_page_config(
        page_title="PragyanAI Ingestion Portal",
        page_icon="📊",
        layout="wide"
    )

    st.title("PragyanAI Enterprise Ingestion & Sanitization Engine")
    st.subheader("Process seat matrix documentation while automatically isolating rows and removing unmapped layout noise.")
    st.markdown("---")

    # 1. Page Header Branding UI Block
    st.title("PragyanAI Enterprise Ingestion & Sanitization Engine")
    st.subheader("Process seat matrix documentation while automatically isolating rows and removing unmapped layout noise.")
    st.markdown("---")

    # 2. Control Layout Panel Widgets
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        st.markdown("#### Horizon Configuration")
        target_year = st.selectbox(
            "Assign Academic Intake Target Year Profile:",
            options=[2024, 2025, 2026, 2027],
            index=2,
            help="Select the exact calendar year to stamp and index these incoming institutional metrics."
        )
        
    with col_w2:
        st.markdown("#### Ingestion Input Mode")
        input_channel = st.radio(
            "Select Document Source Input Stream Channel:",
            options=["Binary File Upload Desktop Interface", "Raw Text Buffer Clipboard Injection Desk"],
            index=0,
            horizontal=True
        )

    st.markdown("---")

    raw_normalized_text_dump = ""
    file_display_name = ""

    # 3. Form Input Method Handling
    if "Binary File Upload Desktop Interface" in input_channel:
        uploaded_file = st.file_uploader(
            f"Drag and drop or browse local system files to clean for Academic Session {target_year}:",
            type=["pdf", "xlsx", "xls", "csv", "docx", "doc"],
            help="Accepts government seat matrix PDF records, compiled program spreadsheets, or word tables."
        )
        
        if uploaded_file is not None:
            file_display_name = uploaded_file.name
            with st.spinner("Streaming binary layout vectors directly to memory buffers..."):
                try:
                    file_binary_payload = uploaded_file.read()
                    # Route to ensemble AI parsing pipeline (Docling, Marker, Unstructured)
                    raw_normalized_text_dump = PragyanDocumentExtractor.extract_to_text_stream(
                        file_binary_payload, 
                        file_display_name
                    )
                except Exception as ex:
                    st.error(f"Ingestion Module Failure: File text layer extraction crashed: {str(ex)}")
                    st.stop()
    else:
        file_display_name = f"Manual_Clipboard_Injection_{target_year}.txt"
        raw_normalized_text_dump = st.text_area(
            f"Paste Raw Unstructured Text String Block Context for Year {target_year} Here:",
            height=300,
            placeholder="Paste text matrix layout here...",
            help="Directly inject copied console characters or unformatted terminal text outputs."
        )

    # 4. Pipeline Execution & Sanitization Trigger
    if st.button(f" Execute Ingestion & Deduplication Pipeline", type="primary", use_container_width=True):
        if not raw_normalized_text_dump.strip():
            st.warning("Execution Halted: No acceptable raw textual context metadata or document streams were found.")
            return

        with st.status("Executing Multi-Stage Data Sanitization Pipeline...", expanded=True) as status_block:
            try:
                # Stage A: Run Regex-LLM parsing structures
                status_block.write(" *Phase [1/3]: Running state-machine layout matching checks...*")
                parser_engine = CETSeatMatrixParser()
                
                # The updated parser engine runs built-in deduplication before returning the DataFrame
                sanitized_df = parser_engine.parse_text_stream(raw_normalized_text_dump, target_year)
                
                # Stage B: Check validation and data density bounds
                status_block.write(" *Phase [2/3]: Filtering unmapped rows and structural text-wraps...*")

                if sanitized_df is not None and not sanitized_df.empty:
                    status_block.write(" *Phase [3/3]: Synchronizing unique records with local SQLite data lake...*")
                    
                    # Commit deduplicated record rows into persistence layers
                    save_matrix_records(sanitized_df)
                    
                    # Close status panel as successfully finalized
                    status_block.update(label="🎉 Relational Matrix Extraction and Deduplication Complete!", state="complete")
                    
                    # 5. Render Post-Extraction Metrics Telemetry Dashboard
                    st.markdown("###  Cleaning & Deduplication Telemetry")
                    
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        # Raw extraction length approximation before unique compression filters
                        st.metric(label="Sanitized Data Matrix Size", value=f"{len(sanitized_df)} Rows")
                    with m_col2:
                        st.metric(label="Unique Campuses Extracted", value=f"{sanitized_df['college_name'].nunique()}")
                    with m_col3:
                        st.metric(label="Total Admitted Capacity Added", value=f"{sanitized_df['intake'].sum():,} Seats")
                        
                    st.success(f"Successfully processed **{file_display_name}** for Academic Year {target_year}! Null fields and layout-echo duplicates were stripped automatically.")
                    
                    # Render sanitized data frame viewport preview
                    with st.expander(" Review Clean relational Database Records", expanded=True):
                        st.dataframe(
                            sanitized_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "college_name": "Institution Entity Name",
                                "city": "City",
                                "district": "District",
                                "address": "Physical Verification Address",
                                "dept": "Sanitized Engineering Discipline Track",
                                "intake": st.column_config.NumberColumn("Allotted Intake", format="%d"),
                                "intake_year": st.column_config.NumberColumn("Target Horizon Year", format="%d")
                            }
                        )
                else:
                    status_block.update(label="❌ Spatial Extraction Signature Violation Encountered.", state="error")
                    st.error("The system failed to extract data rows. Check if your source file aligns with target matrix column signatures.")
                    
            except Exception as pipeline_err:
                status_block.update(label="💥 Ingestion Pipeline Crash Interrupted Execution.", state="error")
                st.error(f"Critical Parsing Exception Intercepted: {str(pipeline_err)}")

if __name__ == "__main__":
    render_ingestion_portal()
