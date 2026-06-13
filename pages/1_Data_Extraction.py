import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import requests
import json
from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# 1. ENVIRONMENT WORKSPACE & MODEL ROUTING
# ==============================================================================
MODEL_DIR = os.path.join(tempfile.gettempdir(), "rapidocr_models")
MODEL_NAME = "ch_PP-OCRv4_det_mobile.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
DOWNLOAD_URL = f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/torch/PP-OCRv4/det/{MODEL_NAME}"

os.environ["RAPIDOCR_MODEL_DIR"] = MODEL_DIR
os.environ["PPOCR_HOME"] = os.path.join(tempfile.gettempdir(), "ppocr_cache")
os.environ["DOCLING_ARTIFACTS_PATH"] = os.path.join(tempfile.gettempdir(), "docling_cache")
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "huggingface_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "xdg_cache")

def ensure_model_exists():
    """Validates presence of model file binary locally; downloads to /tmp if missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        st.info("📢 [LOG - SYSTEM INIT]: Local OCR Engine weight model not found. Fetching now...")
        try:
            response = requests.get(DOWNLOAD_URL, stream=True, timeout=45)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("🛰️ [LOG - SYSTEM INIT]: Writable OCR weights downloaded successfully to /tmp.")
            else:
                st.error(f"❌ [LOG - CRITICAL]: Download failed. HTTP Status Code: {response.status_code}")
        except Exception as e:
            st.error(f"❌ [LOG - CRITICAL]: Network failure pulling OCR assets: {str(e)}")

ensure_model_exists()

# Safe initialization imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 2. ENFORCED TARGET RECONSTRUCTION SCHEMA
# ==========================================
class SeatMatrixRecord(BaseModel):
    college_name: str = Field(..., description="Full clean name of the institution (e.g., 'ACS College of Engineering')")
    city: Optional[str] = Field(None, description="City location name if clearly stated or extracted, otherwise null.")
    district: Optional[str] = Field(None, description="Target district context if clearly stated, otherwise null.")
    address: Optional[str] = Field(None, description="The complete clean street/location address string.")
    dept: Optional[str] = Field(None, description="Academic department course / branch category.")
    intake: Optional[int] = Field(None, description="Total absolute allocated seat matrix capacity value. Convert to integer.")
    intake_year: Optional[int] = Field(None, description="The structural allocation year context for this specific row.")

class SeatMatrixExtraction(BaseModel):
    records: List[SeatMatrixRecord] = Field(default=[], description="Structured institutional rows matching target schema formatting.")

# ==========================================
# 3. DATABASE INSERTION LOG ENGINE
# ==========================================
def save_dataframe_to_sqlite(df: pd.DataFrame, chunk_id: int):
    """Inserts a fragment of the extracted dataframe into the local SQLite database and returns true."""
    st.markdown(f"💾 **[DB ENGINE LOG]: Initiating write transaction for Data Fragment Batch {chunk_id}...**")
    try:
        conn = sqlite3.connect("matrix_records.db")
        df.to_sql("colleges", conn, if_exists="append", index=False)
        conn.commit()
        
        # Verify transaction insertion count
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM colleges")
        total_rows = cursor.fetchone()[0]
        conn.close()
        
        st.success(f"✔️ **[DB ENGINE LOG]: Batch {chunk_id} committed successfully. Total rows now in SQL DB: {total_rows}**")
    except Exception as db_err:
        st.error(f"❌ **[DB ENGINE LOG - ERROR]: Failed to write chunk {chunk_id} to database: {str(db_err)}**")

# ==========================================
# 4. USER INTERFACE LAYOUT PLATFORM
# ==========================================
st.set_page_config(page_title="Data Extraction Engine", layout="wide")

st.header("⚙️ Step-by-Step Interactive Extraction & Relational Sync Pipeline")
st.markdown("""
This dashboard details the document extraction sequence part-by-part. 
Below, you can observe raw inputs, markdown conversions, prompt generations, token responses, and SQL logs.
""")

# Route API credential configuration via secrets safely
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("🔒 Authenticated securely via Streamlit Secrets!")
else:
    st.sidebar.warning("⚠️ GROQ_API_KEY missing from st.secrets configuration.")
    groq_api_key = st.sidebar.text_input("Provide Groq API Key Manually:", type="password", value=os.getenv("GROQ_API_KEY", ""))

target_year = st.number_input(
    "Specify Target Intake Allocation Year:", 
    min_value=2020, max_value=2035, value=2024
)

uploaded_file = st.file_uploader("Upload Seat Matrix Registry PDF File", type=["pdf"])

# ==========================================
# 5. EXECUTION PIPELINE RUNNER
# ==========================================
if st.button("Execute Guided Extraction Pipeline") and uploaded_file:
    if not groq_api_key:
        st.error("❌ Authorization terminated: Missing valid Groq API Key.")
        st.stop()
        
    # Creation of persistent display zones for step logs
    log_status_pane = st.container()
    step1_pane = st.expander("📝 STEP 1 LOGS: Document Layout Extraction (Docling)", expanded=True)
    step2_pane = st.expander("🔀 STEP 2 LOGS: Text Splitting & Fragment Isolation", expanded=False)
    step3_pane = st.expander("🤖 STEP 3 LOGS: LLM Chunk Schema Parsing (Part-by-Part)", expanded=False)
    
    with log_status_pane:
        st.info("🚀 Starting core data pipeline execution framework. Tracking updates below...")

    # --------------------------------------------------------------------------
    # PART 1: DOCLING LAYOUT CONVERSION
    # --------------------------------------------------------------------------
    with step1_pane:
        st.write("⏳ Creating local file streams and spinning up layout parser engines...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        st.code(f"[FILE MANAGER]: Stream intercepted. Temporary location mapped to: {tmp_path}")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False 
        pipeline_options.do_table_structure = True 
        
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        conversion_output = converter.convert(tmp_path)
        markdown_payload = conversion_output.document.export_to_markdown()
        
        st.success("✅ **Docling Layout Stage Complete! Preview of recovered structure below:**")
        st.text_area("Raw Extracted Markdown Snippet (First 800 Characters):", value=markdown_payload[:800], height=200)

    # --------------------------------------------------------------------------
    # PART 2: TEXT SPLITTING
    # --------------------------------------------------------------------------
    with step2_pane:
        st.write("✂️ Initializing Recursive Character text partition layers...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
        text_fragments = splitter.split_text(markdown_payload)
        
        st.success(f"✅ **Text partition completed successfully. Total processing fragments created: {len(text_fragments)}**")
        for idx, frag in enumerate(text_fragments):
            st.code(f"Fragment Index [{idx + 1}]: Character Count: {len(frag)} bytes | Preview text snippet: {frag[:80]}...")

    # --------------------------------------------------------------------------
    # PART 3: LLM CHUNK SCHEMA PARSING & PART-BY-PART DUMP
    # --------------------------------------------------------------------------
    with step3_pane:
        st.write("🤖 Connecting to Groq Inference Mesh layer. Starting chunk parsing iterations...")
        
        llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
        structured_extractor = llm_router.with_structured_output(SeatMatrixExtraction)
        
        prompt_blueprint = ChatPromptTemplate.from_messages([
            ("system", """You are a tabular legal document translation parser. 
            Convert raw engineering seat text tables into accurate, structured row records.
            
            CRITICAL EXTRACTION COMPLIANCE DIRECTIVES:
            1. Identify the 'college_name' precisely. Do not truncate strings.
            2. If address attributes contain tokens like 'BANGALORE' or 'BENGALURU', parse that into the 'city' property.
            3. If an explicit column cell property can't be found anywhere within the text context, return null for that attribute.
            4. Set 'intake_year' explicitly to {year} for all individual records parsed.
            """),
            ("user", "Extract data elements from this segment context block:\n\n{text_fragment}")
        ])
        
        global_record_holder = []
        progress_bar = st.progress(0)
        
        # Loop over every fragment chunk, parse it, and write it directly to the DB
        for step_index, chunk_fragment in enumerate(text_fragments):
            chunk_display_id = step_index + 1
            st.markdown(f"--- \n### 📦 PROCESSING FRAGMENT BLOCK ({chunk_display_id} / {len(text_fragments)})")
            
            # Show input payload log
            with st.expander(f"View Chunk {chunk_display_id} Input Text Passed to Model", expanded=False):
                st.text(chunk_fragment)
            
            formatted_prompt = prompt_blueprint.format_messages(
                year=target_year, 
                text_fragment=chunk_fragment
            )
            
            # Run LLM prediction
            st.write(f"📡 Sending block {chunk_display_id} to Llama-3.3-70b via Groq API...")
            extraction_result = structured_extractor.invoke(formatted_prompt)
            
            chunk_records = []
            if extraction_result and extraction_result.records:
                for record in extraction_result.records:
                    chunk_records.append(record.model_dump())
                    global_record_holder.append(record.model_dump())
                
                st.write(f"✨ Model returned **{len(chunk_records)} records** from block {chunk_display_id}.")
                
                # Turn batch records into a dataframe
                chunk_df = pd.DataFrame(chunk_records)
                
                # Check column consistency
                for col in ["city", "district", "address", "dept", "intake"]:
                    if col not in chunk_df.columns:
                        chunk_df[col] = None
                
                # Reorder columns
                chunk_df = chunk_df[["college_name", "city", "district", "address", "dept", "intake", "intake_year"]]
                
                # Render the data frame snapshot on-screen
                st.dataframe(chunk_df, use_container_width=True)
                
                # Core write operation: Dump data directly into local DB part-by-part
                save_dataframe_to_sqlite(chunk_df, chunk_id=chunk_display_id)
                
            else:
                st.warning(f"⚠️ Model response for block {chunk_display_id} did not yield valid rows. Skipping database write.")
            
            # Update workflow status bars
            progress_bar.progress(chunk_display_id / len(text_fragments))

        # --------------------------------------------------------------------------
        # PIPELINE RUN CONSOLIDATION
        # --------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("## 🏁 FINAL PIPELINE COMPLETE SUMMARY STATUS")
        
        if global_record_holder:
            final_consolidated_df = pd.DataFrame(global_record_holder)
            
            for col in ["city", "district", "address", "dept", "intake"]:
                if col not in final_consolidated_df.columns:
                    final_consolidated_df[col] = None
                    
            final_consolidated_df = final_consolidated_df[[
                "college_name", "city", "district", "address", "dept", "intake", "intake_year"
            ]]
            
            # Update global state manager so other application files can find it
            st.session_state["shared_dataframe"] = final_consolidated_df
            
            st.success(f"🎉 Fully completed document analysis run. Processed {len(final_consolidated_df)} total matrix items across all files.")
            st.dataframe(final_consolidated_df, use_container_width=True)
            
            # Final binary download file exporter
            csv_payload = final_consolidated_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Complete Integrated Matrix Register as CSV File",
                data=csv_payload,
                file_name=f"seat_matrix_complete_log_{target_year}.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Pipeline finished executing, but no records could be successfully compiled across the entire document sequence.")
            
        # Clean up transient files
        os.unlink(tmp_path)
