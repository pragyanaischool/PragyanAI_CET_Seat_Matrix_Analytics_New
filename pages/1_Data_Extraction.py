# pages/1_Data_Extraction.py
import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import pypdf
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ==============================================================================
# 1. ENFORCED RECONSTRUCTION DATA SCHEMA (WITH STRING STANDARDIZATION & NULL DEFENSE)
# ==============================================================================
class SeatMatrixRecord(BaseModel):
    college_name: str = Field(..., description="Full clean name of the institution (e.g., 'ACS College of Engineering' or 'THE KISHKINDA UNIVERSITY')")
    city: Union[str, None] = Field(default="", description="City location name if clearly stated or extracted, otherwise empty string. Do not use literal null.")
    district: Union[str, None] = Field(default="", description="Target district context if clearly stated, otherwise empty string. Do not use literal null.")
    address: Union[str, None] = Field(default="", description="The complete clean street/location address string or empty string. Do not use literal null.")
    dept: Union[str, None] = Field(default="", description="The normalized academic department, course name, or branch name (e.g., 'COMPUTER SCIENCE AND ENGG'). Uniformly capture all variations here. Do not use literal null.")
    intake: Union[int, None] = Field(default=0, description="Total absolute allocated seat matrix capacity value. Convert to integer format.")
    intake_year: int = Field(..., description="The structural allocation year context for this specific row.")

class SeatMatrixExtraction(BaseModel):
    records: List[SeatMatrixRecord] = Field(default_factory=list, description="Structured institutional rows matching target schema formatting.")

# ==========================================
# 2. SEGMENTED DATABASE WRITE TRANSACTION
# ==========================================
def save_dataframe_to_sqlite(df: pd.DataFrame, chunk_id: int):
    """Inserts a fragment of the extracted dataframe into the local SQLite database part-by-part."""
    st.markdown(f"💾 **[DB ENGINE LOG]: Initiating write transaction for Data Fragment Batch {chunk_id}...**")
    try:
        conn = sqlite3.connect("matrix_records.db")
        df.to_sql("colleges", conn, if_exists="append", index=False)
        conn.commit()
        
        # Verify transaction database row checks
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM colleges")
        total_rows = cursor.fetchone()[0]
        conn.close()
        
        st.success(f"✔️ **[DB ENGINE LOG]: Batch {chunk_id} committed successfully. Total rows now in SQL DB: {total_rows}**")
    except Exception as db_err:
        st.error(f"❌ **[DB ENGINE LOG - ERROR]: Failed to write chunk {chunk_id} to database: {str(db_err)}**")

# ==========================================
# 3. USER INTERFACE LAYOUT PLATFORM
# ==========================================
st.set_page_config(page_title="Data Extraction Engine", layout="wide")

st.header("⚙️ Step-by-Step Interactive Extraction & Relational Sync Pipeline")
st.markdown("""
This module extracts tabular schedules directly from your files step-by-step. 
By pulling text streams locally via zero-weight backends, we completely sidestep system rendering issues (`libGL.so.1` or missing `.safetensors`).
""")

# Route API credential configuration via secrets safely
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("🔒 Authenticated securely via Streamlit Secrets!")
else:
    st.sidebar.warning("⚠️ GROQ_API_KEY missing from st.secrets configuration.")
    groq_api_key = st.sidebar.text_input("Provide Groq API Key Manually:", type="password", value=os.getenv("GROQ_API_KEY", ""))

# ==========================================
# MULTI-LLM SELECTION ROUTER PANEL (GROQ ONLY)
# ==========================================
st.sidebar.subheader("🎯 Model Configuration Hub")
model_selection = st.sidebar.selectbox(
    "Select Target Execution LLM Engine:",
    [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b"
    ],
    index=2  # Default to llama-3.1-8b-instant
)

# Explicitly map user model selections to native Groq endpoints
if model_selection == "openai/gpt-oss-120b":
    groq_model_id = "llama-3.3-70b-versatile"
    st.sidebar.caption("🚀 Running via high-reasoning Llama-3.3-70b architecture")
elif model_selection == "openai/gpt-oss-20b":
    groq_model_id = "gemma2-9b-it"
    st.sidebar.caption("⚡ Running via efficient instruction-tuned Gemma dense layer")
elif model_selection == "llama-3.1-8b-instant":
    groq_model_id = "llama-3.1-8b-instant"
    st.sidebar.caption("🏎️ Running via ultra-low latency native 8B framework")
elif model_selection == "qwen/qwen3-32b":
    groq_model_id = "qwen-2.5-32b"
    st.sidebar.caption("📊 Running via high-compliance Qwen structure matrix matrix")

target_year = st.number_input(
    "Specify Target Intake Allocation Year:", 
    min_value=2020, max_value=2035, value=2024
)

uploaded_file = st.file_uploader("Upload Seat Matrix Registry PDF File", type=["pdf"])

# ==========================================
# 4. EXECUTION PIPELINE RUNNER
# ==========================================
if st.button("Execute Guided Extraction Pipeline") and uploaded_file:
    if not groq_api_key:
        st.error("❌ Authorization terminated: Missing valid Groq API Key.")
        st.stop()
        
    log_status_pane = st.container()
    step1_pane = st.expander("📝 STEP 1 LOGS: Local Stream Text Extraction (Zero-Weight Engine)", expanded=True)
    step2_pane = st.expander("🔀 STEP 2 LOGS: Text Splitting & Fragment Isolation", expanded=False)
    step3_pane = st.expander("🤖 STEP 3 LOGS: LLM Chunk Schema Parsing & DB Dump (Part-by-Part)", expanded=True)
    
    with log_status_pane:
        st.info("🚀 Starting core data pipeline execution framework. Tracking updates below...")

    # --------------------------------------------------------------------------
    # PART 1: ZERO-WEIGHT LOCAL STREAM RECOVERY
    # --------------------------------------------------------------------------
    with step1_pane:
        st.write("⏳ Intercepting uploaded asset streams and forcing local text stream parser...")
        
        try:
            # Parse the PDF text layer using a pure Python stream memory processor
            pdf_reader = pypdf.PdfReader(uploaded_file)
            extracted_lines = []
            
            for page_idx, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_lines.append(f"--- PAGE {page_idx + 1} ---\n{page_text}")
            
            markdown_payload = "\n".join(extracted_lines)
            
            st.success("✅ **Text Stream Extraction Complete! Text preview below:**")
            st.text_area("Raw Extracted Text Stream (First 1000 Characters):", value=markdown_payload[:1000], height=200)
        except Exception as parse_error:
            st.error(f"❌ Failed local processing stream extraction: {str(parse_error)}")
            st.stop()

    # --------------------------------------------------------------------------
    # PART 2: TEXT SPLITTING
    # --------------------------------------------------------------------------
    with step2_pane:
        st.write("✂️ Initializing structural token partition boundaries...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        text_fragments = splitter.split_text(markdown_payload)
        
        st.success(f"✅ **Text partition completed successfully. Total fragments generated: {len(text_fragments)}**")
        for idx, frag in enumerate(text_fragments):
            st.code(f"Fragment Index [{idx + 1}]: Character Count: {len(frag)} characters | Preview: {frag[:80].strip()}...")

    # --------------------------------------------------------------------------
    # PART 3: LLM CHUNK SCHEMA PARSING & PART-BY-PART DUMP
    # --------------------------------------------------------------------------
    with step3_pane:
        st.write("🤖 Instantiating schema structural extractors via Groq API. Starting part-by-part loops...")
        
        # Base setup for the extraction template configuration prompt layout rules
        prompt_blueprint = ChatPromptTemplate.from_messages([
            ("system", """You are an expert tabular data translation engine. 
            Convert raw engineering seat matrices, schedules, lines, and blocks into precise data models matching the target structural schema format.
            
            CRITICAL SYNONYM NORMALIZATION DIRECTIVES:
            1. The text input will alternate column headings interchangeably using terms like 'Course Name', 'Branch Name', 'branch', or 'dept'. 
            2. You MUST treat 'Course Name', 'Branch Name', 'branch', and 'dept' as completely identical concepts. 
            3. Normalize and map ALL of those discovered parameters exclusively into the unified schema field named `dept` (e.g., whether it reads 'Course Name: COMPUTER SCIENCE AND ENGG' or '5 B TECH IN MECHANICAL ENGINEERING', map it straight into the `dept` field property).
            
            CRITICAL DESIGN & VALIDATION SAFETY DIRECTIVES:
            1. Extract the 'college_name' precisely. Do not truncate strings. (e.g., 'THE KISHKINDA UNIVERSITY').
            2. If structural address contexts mention items like 'BANGALORE', 'BENGALURU', or 'BELLARI', map that data directly to the corresponding 'city' or 'district' fields.
            3. CRITICAL DEFENSE AGAINST 400 ERRORS: If an explicit column cell property or attribute value cannot be found anywhere within the provided text block fragment, you MUST output an empty string ("") for that field.
            4. NEVER output a literal JSON `null` value for any key parameter under any circumstance. This violates the API tool-use validator and crashes the generation loop.
            5. Force 'intake_year' explicitly to {year} for every single individual record parsed.
            """),
            ("user", "Convert data attributes cleanly from this document context block:\n\n{text_fragment}")
        ])
        
        global_record_holder = []
        progress_bar = st.progress(0)
        
        # Sequentially scan each text block, map schemas, and commit instantly to database
        for step_index, chunk_fragment in enumerate(text_fragments):
            chunk_display_id = step_index + 1
            st.markdown(f"--- \n### 📦 PROCESSING FRAGMENT BLOCK ({chunk_display_id} / {len(text_fragments)})")
            
            with st.expander(f"View Chunk {chunk_display_id} Raw Text Token Flow Input", expanded=False):
                st.text(chunk_fragment)
            
            formatted_prompt = prompt_blueprint.format_messages(
                year=target_year, 
                text_fragment=chunk_fragment
            )
            
            # Dynamic Model Rotation Selector Check Engine
            # Resolves potential stability errors on high-fatigue or complex table blocks dynamically
            active_model_id = groq_model_id
            if chunk_display_id >= 8 and groq_model_id in ["llama-3.1-8b-instant", "gemma2-9b-it"]:
                st.caption("🔄 *[PIPELINE OPTIMIZATION]*: Switching tracking over to high-reasoning flagship layer `llama-3.3-70b-versatile` to reinforce table compliance across processing thresholds...")
                active_model_id = "llama-3.3-70b-versatile"

            st.write(f"📡 Dispatching block {chunk_display_id} to Groq engine execution path via `{active_model_id}`...")
            
            try:
                llm_router = ChatGroq(model=active_model_id, temperature=0, api_key=groq_api_key)
                structured_extractor = llm_router.with_structured_output(SeatMatrixExtraction)
                
                extraction_result = structured_extractor.invoke(formatted_prompt)
                
                chunk_records = []
                if extraction_result and extraction_result.records:
                    for record in extraction_result.records:
                        chunk_records.append(record.model_dump())
                        global_record_holder.append(record.model_dump())
                    
                    st.write(f"✨ Model successfully isolated **{len(chunk_records)} structured data rows** from block {chunk_display_id}.")
                    
                    # Turn batch records into a unified data frame instance
                    chunk_df = pd.DataFrame(chunk_records)
                    
                    # Post-processing layer validation: Ensure missing variables perfectly fill as standard strings
                    for col in ["city", "district", "address", "dept", "intake"]:
                        if col not in chunk_df.columns:
                            chunk_df[col] = ""
                        else:
                            # Catch and remove any lingering null structures safely
                            chunk_df[col] = chunk_df[col].fillna("")
                    
                    # Re-align index positions explicitly
                    chunk_df = chunk_df[["college_name", "city", "district", "address", "dept", "intake", "intake_year"]]
                    
                    # Display the data segment snapshot instantly on screen
                    st.dataframe(chunk_df, use_container_width=True)
                    
                    # Real-time part-by-part write straight to local database file
                    save_dataframe_to_sqlite(chunk_df, chunk_id=chunk_display_id)
                else:
                    st.warning(f"⚠️ Block {chunk_display_id} did not yield clean row models matching validations. Skipping write transaction.")
            except Exception as invoke_error:
                st.error(f"❌ Failed inference tracking execution on Block {chunk_display_id}: {str(invoke_error)}")
                st.info("💡 Pro Tip: The pipeline will continue processing subsequent blocks to prevent losing previous ingestion tracking work.")
            
            progress_bar.progress(chunk_display_id / len(text_fragments))

        # --------------------------------------------------------------------------
        # PIPELINE RUN CONSOLIDATION & DYNAMIC FILTER LOOKUP VIEW PANEL
        # --------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("## 🏁 PIPELINE RECONSTRUCTION CONSOLIDATION STATUS")
        
        if global_record_holder:
            final_consolidated_df = pd.DataFrame(global_record_holder)
            
            for col in ["city", "district", "address", "dept", "intake"]:
                if col not in final_consolidated_df.columns:
                    final_consolidated_df[col] = ""
                else:
                    final_consolidated_df[col] = final_consolidated_df[col].fillna("")
                    
            final_consolidated_df = final_consolidated_df[[
                "college_name", "city", "district", "address", "dept", "intake", "intake_year"
            ]]
            
            # Commit the unified frame into global memory state shared across app screens
            st.session_state["shared_dataframe"] = final_consolidated_df
            
            st.success(f"🎉 Fully completed document analysis run. Processed {len(final_consolidated_df)} total matrix items across all files.")
            st.dataframe(final_consolidated_df, use_container_width=True)
            
            csv_payload = final_consolidated_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Integrated Matrix Register as CSV File",
                data=csv_payload,
                file_name=f"seat_matrix_complete_log_{target_year}.csv",
                mime="text/csv"
            )
            
            # ==================================================================
            # 5. DYNAMIC FILTER BASED VIEW MODULE PANEL
            # ==================================================================
            st.markdown("---")
            st.markdown("### 🔍 Live Extraction Register Filter Console")
            st.markdown("Apply parameters to segment the raw in-memory records below on-the-fly:")
            
            # Isolate list constraints dynamically from currently extracted data rows
            unique_cities = sorted(list(set([str(c).strip() for c in final_consolidated_df["city"].unique() if c])))
            unique_branches = sorted(list(set([str(b).strip() for b in final_consolidated_df["dept"].unique() if b])))
            
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                sel_cities = st.multiselect("Filter On-The-Fly View by City Location:", options=unique_cities, default=unique_cities[:2] if unique_cities else [])
            with f_col2:
                sel_branches = st.multiselect("Filter On-The-Fly View by Normalized Departments / Course Streams:", options=unique_branches, default=unique_branches if unique_branches else [])
                
            # Render sliced snapshot view based on interactive dashboard conditions
            if sel_cities or sel_branches:
                query_view_df = final_consolidated_df.copy()
                if sel_cities:
                    query_view_df = query_view_df[query_view_df["city"].isin(sel_cities)]
                if sel_branches:
                    query_view_df = query_view_df[query_view_df["dept"].isin(sel_branches)]
                    
                st.markdown(f"Displaying **{len(query_view_df)} entries** inside filter grid view:")
                st.dataframe(query_view_df, use_container_width=True)
            else:
                st.info("Select city or branch targets inside the lookup box fields above to populate matching entries instantly.")
            
        else:
            st.error("❌ Pipeline finished executing, but zero rows successfully passed data schema filters across the full document context.")
            
