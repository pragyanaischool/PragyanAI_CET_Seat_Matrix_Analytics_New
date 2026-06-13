import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
from typing import List, Optional
from pydantic import BaseModel, Field

# ==============================================================================
# 0. STRICT ISOLATION OF GLOBAL CACHE SYSTEM PATHS (SYSTEM ENVIRONMENT LEVEL)
# ==============================================================================
tmp_dir = tempfile.gettempdir()
os.environ["RAPIDOCR_MODEL_DIR"] = os.path.join(tmp_dir, "rapidocr_models")
os.environ["HF_HOME"] = os.path.join(tmp_dir, "huggingface_cache")
os.environ["XDG_CACHE_HOME"] = os.path.join(tmp_dir, "xdg_cache")
os.environ["TORCH_HOME"] = os.path.join(tmp_dir, "torch_cache")

# Build target operational folders programmatically to ensure write permissions
os.makedirs(os.environ["RAPIDOCR_MODEL_DIR"], exist_ok=True)

# Safe to import core structural data pipeline modules now
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. ENFORCED TARGET SCHEMA CONFIGURATION
# ==========================================
class SeatMatrixRecord(BaseModel):
    college_name: str = Field(..., description="Full clean name of the institution (e.g., 'ACS College of Engineering' or 'THE KISHKINDA UNIVERSITY')")
    city: Optional[str] = Field(None, description="City location name if clearly stated or extracted, otherwise null.")
    district: Optional[str] = Field(None, description="Target district context if clearly stated, otherwise null.")
    address: Optional[str] = Field(None, description="The complete clean street/location address string.")
    dept: Optional[str] = Field(None, description="Academic department course / branch category.")
    intake: Optional[int] = Field(None, description="Total absolute allocated seat matrix capacity value. Convert to integer.")
    intake_year: Optional[int] = Field(None, description="The structural allocation year context for this specific row.")

class SeatMatrixExtraction(BaseModel):
    records: List[SeatMatrixRecord] = Field(default=[], description="Structured institutional rows matching target schema formatting.")

# ==========================================
# 2. DATABASE RELATIONAL OPERATION LAYER
# ==========================================
def save_dataframe_to_sqlite(df: pd.DataFrame):
    """Inserts the extracted data frame into the local shared SQLite system."""
    conn = sqlite3.connect("matrix_records.db")
    df.to_sql("colleges", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

# ==========================================
# 3. USER INTERFACE LAYOUT PLATFORM
# ==========================================
st.set_page_config(page_title="Data Extraction Engine", layout="wide")

st.header("⚙️ Document Extraction & Relational Sync Matrix Engine")
st.markdown("""
This module extracts layout-heavy tabular data from engineering seat matrices, normalizes attributes to schema-rigid configurations, and logs records to SQL.
""")

# ==========================================
# FETCHING GROQ API KEY FROM STREAMLIT SECRETS
# ==========================================
# Check secrets workspace first, fall back to environment variables or manual fallback strings
if "GROQ_API_KEY" in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("🔒 Authenticated securely via Streamlit Secrets!")
else:
    st.sidebar.warning("⚠️ GROQ_API_KEY missing from st.secrets")
    groq_api_key = st.sidebar.text_input("Provide Groq API Key Manually:", type="password", value=os.getenv("GROQ_API_KEY", ""))

target_year = st.number_input(
    "Specify Target Intake Allocation Year:", 
    min_value=2020, 
    max_value=2035, 
    value=2024,
    help="This value populates the intake_year attribute column for every entry tuple."
)

uploaded_file = st.file_uploader("Upload Seat Matrix Registry PDF File", type=["pdf"])

# ==========================================
# 4. EXECUTION PIPELINE RUNNER
# ==========================================
if st.button("Execute Extraction & Sync Pipeline") and uploaded_file:
    if not groq_api_key:
        st.error("❌ Authorization terminated: Please enter or provide a valid Groq API Key.")
        st.stop()
        
    with st.spinner("Step 1: Instantiating Docling Layout Parsers with Safe Defaults..."):
        try:
            # Trap the uploaded stream footprint inside transient volume paths safely
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # ==================================================================
            # FIX: REMOVED BLANK OcrOptions() VALUE INITIALIZATION TO PREVENT PYDANTIC ERROR
            # ==================================================================
            # Allowing PdfPipelineOptions to spin up its native default settings avoids missing 'lang' errors
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True  # Explicitly enable text layer recovery securely
            
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            conversion_output = converter.convert(tmp_path)
            markdown_payload = conversion_output.document.export_to_markdown()
            
            st.success("✅ Layout structural tables indexed safely using default writable option parameters!")
            
            # Subdivide heavy text components into processing segments
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
            text_fragments = splitter.split_text(markdown_payload)
            
            # Connect LangChain mapping instances to Groq Mesh
            st.caption("🔄 Spawning schema-enforced processing workers over Groq Inference Mesh...")
            llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
            structured_extractor = llm_router.with_structured_output(SeatMatrixExtraction)
            
            prompt_blueprint = ChatPromptTemplate.from_messages([
                ("system", """You are a tabular document translation parser. 
                Convert raw engineering seat text grids into clear structured records.
                
                CRITICAL EXTRACTION DIRECTIVES:
                1. Identify 'college_name' precisely. Do not truncate strings.
                2. If address attributes mention tokens like 'BANGALORE' or 'BENGALORE', map that cleanly to the city property.
                3. If an explicit column cell property can't be found anywhere within the text context, return null for that attribute.
                4. Set 'intake_year' explicitly to {year} for all individual records parsed.
                """),
                ("user", "Extract data elements from this segment context block:\n\n{text_fragment}")
            ])
            
            compiled_record_list = []
            progress_indicator = st.progress(0)
            
            for step_index, chunk_fragment in enumerate(text_fragments):
                st.caption(f"Processing structural layout matrix chunk {step_index + 1} / {len(text_fragments)}...")
                
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
                
                # Force alignment to ensure missing property indices display as NaN / None
                for target_column in ["city", "district", "address", "dept", "intake"]:
                    if target_column not in final_extracted_df.columns:
                        final_extracted_df[target_column] = None
                
                # Strict structural index configuration sorting matching layout requirements
                final_extracted_df = final_extracted_df[[
                    "college_name", "city", "district", "address", "dept", "intake", "intake_year"
                ]]
                
                # Save data directly down into relational table architectures
                save_dataframe_to_sqlite(final_extracted_df)
                
                # Sync dataframe parameters cleanly across global application states
                st.session_state["shared_dataframe"] = final_extracted_df
                
                st.success(f"🎉 Pipeline execution success! Synced {len(final_extracted_df)} rows directly into 'matrix_records.db'.")
                st.dataframe(final_extracted_df, use_container_width=True)
                
                # Expose local download handlers
                csv_download_payload = final_extracted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Extracted Matrix Register as CSV",
                    data=csv_download_payload,
                    file_name=f"seat_matrix_extracted_{target_year}.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠️ Document analysis run complete, but zero rows matched the required structural target criteria.")
                
            os.unlink(tmp_path)
            
        except Exception as system_error:
            st.error(f"💥 Processing Engine Core Error Interruption: {str(system_error)}")
