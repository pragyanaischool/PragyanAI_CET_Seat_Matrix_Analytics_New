import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import requests
from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# 1. HARDLOCK WRITABLE DIRECTORY PARAMS & ENVIRONMENT CACHES
# ==============================================================================
MODEL_DIR = os.path.join(tempfile.gettempdir(), "rapidocr_models")
MODEL_NAME = "ch_PP-OCRv4_det_mobile.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

DOWNLOAD_URL = f"https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/torch/PP-OCRv4/det/{MODEL_NAME}"

# Override every possible background artifact route to point to /tmp
os.environ["RAPIDOCR_MODEL_DIR"] = MODEL_DIR
os.environ["PPOCR_HOME"] = os.path.join(tempfile.gettempdir(), "ppocr_cache")
os.environ["DOCLING_ARTIFACTS_PATH"] = os.path.join(tempfile.gettempdir(), "docling_cache")
os.environ["HF_HOME"] = os.path.join(tempfile.gettempdir(), "huggingface_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(tempfile.gettempdir(), "xdg_cache")

def ensure_model_exists():
    """Validates presence of model file binary locally; downloads to /tmp if missing."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with st.spinner("📥 Downloading OCR layout weights to secure writable runtime cache..."):
            try:
                response = requests.get(DOWNLOAD_URL, stream=True, timeout=45)
                if response.status_code == 200:
                    with open(MODEL_PATH, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    st.success("✅ Writable OCR model weights cached successfully!")
                else:
                    st.error(f"❌ Weight provider returned non-200 status: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Network failure pulling OCR assets: {str(e)}")

# Invoke the asset downloader before resolving layout-engine dependencies
ensure_model_exists()

# Safe to bring in processing framework modules now
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 2. ENFORCED RELATIONAL SCHEMA SETUP
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

def save_dataframe_to_sqlite(df: pd.DataFrame):
    """Inserts the extracted dataframe into the local shared SQLite system."""
    conn = sqlite3.connect("matrix_records.db")
    df.to_sql("colleges", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

# ==========================================
# 3. USER INTERFACE COMPONENT PORTAL
# ==========================================
st.set_page_config(page_title="Data Extraction Engine", layout="wide")

st.header("⚙️ Writable Layout Document Extraction & Relational Sync Engine")
st.markdown("""
This module extracts layout-heavy tabular columns using local transient models, structures the layout via LLM schemas, and logs outputs directly to SQLite.
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
# 4. EXECUTION PIPELINE RUNNER
# ==========================================
if st.button("Execute Extraction & Sync Pipeline") and uploaded_file:
    if not groq_api_key:
        st.error("❌ Authorization terminated: Missing a valid Groq API key credential identifier.")
        st.stop()
        
    with st.spinner("Step 1: Instantiating Docling Layout Parsers with Safe Sandbox Contexts..."):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # ==================================================================
            # BYPASS ALL SITE-PACKAGES DOWNLOAD ATTEMPTS AT THE INSTANTIATION LAYER
            # ==================================================================
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False # Disable layout engine tracking to let Docling rely entirely on native PDF text extraction boundaries
            pipeline_options.do_table_structure = True # Focus on building robust text boundaries markdown grids internally
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            conversion_output = converter.convert(tmp_path)
            markdown_payload = conversion_output.document.export_to_markdown()
            
            st.success("✅ Layout structural elements indexed cleanly utilizing safe path models!")
            
            # Split parsed document fragments down to frame matrices
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
            text_fragments = splitter.split_text(markdown_payload)
            
            # Connect LangChain mapping schema interfaces down to Groq endpoints
            st.caption("🔄 Mapping data schemas across active Groq endpoints...")
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
            
            compiled_record_list = []
            progress_indicator = st.progress(0)
            
            for step_index, chunk_fragment in enumerate(text_fragments):
                st.caption(f"Processing structural data fragment {step_index + 1} / {len(text_fragments)}...")
                
                formatted_prompt = prompt_blueprint.format_messages(
                    year=target_year, 
                    text_fragment=chunk_fragment
                )
                
                extraction_result = structured_extractor.invoke(formatted_prompt)
                
                if extraction_result and extraction_result.records:
                    for record in extraction_result.records:
                        compiled_record_list.append(record.model_dump())
                        
                progress_indicator.progress((step_index + 1) / len(text_fragments))
            
            # ==========================================
            # 5. POST-PROCESSING SCHEMA FORMATTING
            # ==========================================
            if compiled_record_list:
                final_extracted_df = pd.DataFrame(compiled_record_list)
                
                # Align parameters to handle empty entries as standard visual NaN attributes
                for target_column in ["city", "district", "address", "dept", "intake"]:
                    if target_column not in final_extracted_df.columns:
                        final_extracted_df[target_column] = None
                
                # Sort explicitly to enforce required column order
                final_extracted_df = final_extracted_df[[
                    "college_name", "city", "district", "address", "dept", "intake", "intake_year"
                ]]
                
                # Commit processed data frame records into persistent DB architecture
                save_dataframe_to_sqlite(final_extracted_df)
                st.session_state["shared_dataframe"] = final_extracted_df
                
                st.success(f"🎉 Pipeline execution success! Synced {len(final_extracted_df)} records directly into local storage matrix.")
                st.dataframe(final_extracted_df, use_container_width=True)
                
                # Expose downloadable CSV binary array stream
                csv_download_payload = final_extracted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Extracted Matrix Register as CSV",
                    data=csv_download_payload,
                    file_name=f"seat_matrix_extracted_{target_year}.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠️ Extraction complete, but zero entries aligned with parsing schema validations.")
                
            os.unlink(tmp_path)
            
        except Exception as system_error:
            st.error(f"💥 Processing Engine Core Error Interruption: {str(system_error)}")
