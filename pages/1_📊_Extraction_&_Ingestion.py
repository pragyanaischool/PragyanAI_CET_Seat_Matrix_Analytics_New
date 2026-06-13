import streamlit as st
from engines.document_extractor import PragyanDocumentExtractor
from engines.matrix_parser import CETSeatMatrixParser
from database.db_handler import save_matrix_records

def render_ingestion_portal():
    """
    Renders the multi-format, year-aware data ingestion and layout 
    healing extraction workspace panel.
    """
    st.set_page_config(
        page_title="PragyanAI Ingestion Portal",
        page_icon="📊",
        layout="wide"
    )

    # 1. Page Section Identification Header
    st.title("📊 Multi-Format Document Ingestion & Extraction Engine")
    st.subheader("Ingest multi-year seats allocations streams from PDF, Word documents, Excel workbooks, or CSV layers.")
    st.markdown("---")

    # 2. Workspace Control Widgets Panel
    col_w1, col_w2 = st.columns([1, 2])
    
    with col_w1:
        st.markdown("#### 📅 Horizon Settings")
        # Explicit drop-down configuration allows mapping input profiles to specific financial cycles
        target_year = st.selectbox(
            "Assign Academic Intake Target Year Profile:",
            options=[2024, 2025, 2026, 2027],
            index=2,
            help="Select the explicit intake calendar year to stamp and index these incoming institutional metrics."
        )
        
    with col_w2:
        st.markdown("#### 🔌 Ingestion Settings")
        input_channel = st.radio(
            "Select Target Document Source Input Stream Channel:",
            options=["Binary File Upload Desktop Interface", "Raw Text Buffer Clipboard Injection Desk"],
            index=0,
            horizontal=True
        )

    st.markdown("---")

    # 3. Form Input Strategy Routing Block
    raw_normalized_text_dump = ""
    file_display_name = ""

    if "Binary File Upload Desktop Interface" in input_channel:
        uploaded_file = st.file_uploader(
            f"Drag and drop or browse local system files to parse for Academic Session {target_year}:",
            type=["pdf", "xlsx", "xls", "csv", "docx", "doc"],
            help="Accepts government seat matrix PDF records, compiled program spreadsheets, or word tables."
        )
        
        if uploaded_file is not None:
            file_display_name = uploaded_file.name
            with st.spinner("Streaming binary layout vectors directly to memory buffers..."):
                try:
                    file_binary_payload = uploaded_file.read()
                    # Route file bytes to the format decoupling abstraction layer
                    raw_normalized_text_dump = PragyanDocumentExtractor.extract_to_text_stream(
                        file_binary_payload, 
                        file_display_name
                    )
                except Exception as ex:
                    st.error(f"Ingestion Module Failure: File content layer could not break down: {str(ex)}")
                    st.stop()
    else:
        file_display_name = f"Manual_Clipboard_Injection_{target_year}.txt"
        raw_normalized_text_dump = st.text_area(
            f"Paste Raw Copied Text String Block Context for Year {target_year} Here:",
            height=350,
            placeholder="Example Layout Profile Pattern:\n1 Government Engineering College, Challakere, Chitradurga\nAddress : BALLARI ROAD CHALLAKERE,CHITRADURGA\nSl.No. Course Name Total Intake...\n1 ARTIFICIAL INTELLIGENCE AND MACHINE LEARNING 60",
            help="Directly inject copied console characters or unformatted terminal text outputs."
        )

    # 4. Pipeline Execution Trigger Desk
    if st.button(f"🚀 Execute Core Extraction Pipeline for Year {target_year}", type="primary", use_container_width=True):
        if not raw_normalized_text_dump.strip():
            st.warning("Execution Halted: No acceptable raw textual context metadata or document streams were found.")
            return

        # Initialize the transaction notification elements inside the UI space
        progress_bar = st.progress(0)
        status_message = st.empty()

        try:
            # Step A: Trigger the state-machine heuristic parsing matrices
            status_message.markdown("🔄 *Phase [1/3]: Running regex parsing state alignments...*")
            progress_bar.progress(30)
            
            parser_engine = CETSeatMatrixParser()
            # The parser accepts the year parameter directly to construct relational rows accurately
            parsed_analytics_df = parser_engine.parse_text_stream(raw_normalized_text_dump, target_year)
            
            # Step B: Evaluate parsed row integrity metrics before storage operations
            status_message.markdown("🔮 *Phase [2/3]: Healing layout artifacts via low-temperature LLM fallbacks...*")
            progress_bar.progress(70)

            if parsed_analytics_df is not None and not parsed_analytics_df.empty:
                status_message.markdown("💾 *Phase [3/3]: Synchronizing structural arrays with relational data lake...*")
                progress_bar.progress(90)
                
                # Commit clean Pandas rows to the SQLite backend database handler
                save_matrix_records(parsed_analytics_df)
                
                # Progress complete updates
                progress_bar.progress(100)
                status_message.empty()
                
                # Render success diagnostic cards directly to the web view layout
                st.success(f"🎉 Pipeline Execution Successful! Decoupled and committed {len(parsed_analytics_df)} row metrics securely to the {target_year} ledger table.")
                
                # Display clean overview data frame metrics window
                with st.expander("👁️ Review Extracted Structural Database Frame Schema Output", expanded=True):
                    st.dataframe(
                        parsed_analytics_df.style.format({"intake": "{:,.0f}", "intake_year": "{:,.0f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                progress_bar.empty()
                status_message.empty()
                st.error("❌ Spatial Extraction Signature Violation: The layout engine completed execution but failed to slice clean relational records. Verify structural format signatures.")
                
        except Exception as pipeline_err:
            progress_bar.empty()
            status_message.empty()
            st.error(f"💥 Critical Ingestion Pipeline Exception Intercepted: {str(pipeline_err)}")

if __name__ == "__main__":
    render_ingestion_portal()
