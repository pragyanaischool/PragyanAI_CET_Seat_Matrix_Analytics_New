import streamlit as st
import pandas as pd
import sqlite3
import os
import tempfile
import re
from typing import List, Optional
from pydantic import BaseModel, Field
from docling.document_converter import DocumentConverter
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
    address: Optional[str] = Field(None, description="The complete clean street/location address string (e.g., '207, KAMBIPURA, MYSORE ROAD, KENGERI HOBLI, BANGALORE - 560074')")
    dept: Optional[str] = Field(None, description="Academic department course / branch category (e.g., 'COMPUTER SCIENCE AND ENGG')")
    intake: Optional[int] = Field(None, description="Total absolute allocated seat matrix capacity value. Convert to integer format.")
    intake_year: Optional[int] = Field(None, description="The structural allocation year context for this specific row.")

class SeatMatrixExtraction(BaseModel):
    records: List[SeatMatrixRecord] = Field(default=[], description="Structured institutional row items tracking index records matching the target schema.")

# ==========================================
# 2. DATABASE SYSTEM OPERATION LAYER
# ==========================================
def save_dataframe_to_sqlite(df: pd.DataFrame):
    """Inserts the extracted data frame into the local shared SQLite system."""
    conn = sqlite3.connect("matrix_records.db")
    
    # We load it using append mode so that historical extractions are retained
    # If columns contain missing values, pandas maps them cleanly as NULL parameters
    df.to_sql("colleges", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()

# ==========================================
# 3. INTERFACE COMPONENT LAYER
# ==========================================
st.set_page_config(page_title="Data Extraction Engine", layout="wide")

st.header("⚙️ Layout-Aware Document Extraction & Database Sync Engine")
st.markdown("""
This module extracts structural multi-column tables from complex seat allocation documents.
It preserves layout parameters using an enterprise markdown analyzer, structures chunks with an LLM parser, and logs results directly into your local database ecosystem.
""")

# Input configuration controllers
col_left, col_right = st.columns(2)
with col_left:
    target_year = st.number_input(
        "Specify Target Intake Allocation Year:", 
        min_value=2020, 
        max_value=2035, 
        value=2024,
        help="This value will populate the intake_year column parameter for every extracted entity row."
    )
with col_right:
    groq_api_key = st.text_input(
        "Groq API Key Authorization:", 
        type="password", 
        value=os.getenv("GROQ_API_KEY", ""),
        help="Required to connect to the LangChain orchestration endpoints."
    )

uploaded_file = st.file_uploader("Upload Seat Matrix Registry PDF File", type=["pdf"])

# ==========================================
# 4. EXECUTION PIPELINE RUNNER
# ==========================================
if st.button("Execute Extraction & Sync Pipeline") and uploaded_file:
    if not groq_api_key:
        st.error("❌ Authorization terminated: Please enter a valid Groq API Key to authenticate inference routing channels.")
        st.stop()
        
    with st.spinner("Step 1: Parsing multi-column document tables using Docling Layout Converter..."):
        try:
            # Safely catch memory stream parameters in a temporary disk location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            # Run layout-aware raw document extraction 
            converter = DocumentConverter()
            conversion_output = converter.convert(tmp_path)
            markdown_payload = conversion_output.document.export_to_markdown()
            
            st.success("✅ Layout structural elements indexed cleanly into markdown format!")
            
            # Context-splitting text into processing chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
            text_fragments = splitter.split_text(markdown_payload)
            
            # Setup Structured LLM Mapping Engine using Groq Architecture
            st.caption("🔄 Spawning LangChain schema-enforced processing workers over Groq Matrix Engine...")
            llm_router = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=groq_api_key)
            structured_extractor = llm_router.with_structured_output(SeatMatrixExtraction)
            
            # Compile target operational instructions prompt
            prompt_blueprint = ChatPromptTemplate.from_messages([
                ("system", """You are a highly precise legal/tabular document translation parser. 
                Your task is to convert raw engineering seat schedules, index matrices, and text lines into clear structured records.
                
                CRITICAL EXTRACTION COMPLIANCE DIRECTIVES:
                1. Identify the 'college_name' precisely. Do not alter abbreviations.
                2. If the address sequence mentions parameters like 'BANGALORE' or 'BELLARI', try to capture the corresponding city/district entries. 
                3. If a property field cannot be located anywhere within the local text fragment block, explicitly yield null for that property parameter.
                4. For each individual record parsed, force the 'intake_year' property strictly to {year}.
                """),
                ("user", "Extract data elements clearly from this document segment context block:\n\n{text_fragment}")
            ])
            
            compiled_record_list = []
            progress_indicator = st.progress(0)
            
            # Iterate and step through document fragments sequentially
            for step_index, chunk_fragment in enumerate(text_fragments):
                st.caption(f"Processing text layout matrix fragment chunk {step_index + 1} / {len(text_fragments)}...")
                
                formatted_prompt = prompt_blueprint.format_messages(
                    year=target_year, 
                    text_fragment=chunk_fragment
                )
                
                # Invoke structured extraction
                extraction_result = structured_extractor.invoke(formatted_prompt)
                
                if extraction_result and extraction_result.records:
                    for record in extraction_result.records:
                        compiled_record_list.append(record.model_dump())
                        
                # Increment screen layout updates smoothly
                progress_indicator.progress((step_index + 1) / len(text_fragments))
            
            # ==========================================
            # 5. POST-PROCESSING SCHEMA FORMATTING
            # ==========================================
            if compiled_record_list:
                final_extracted_df = pd.DataFrame(compiled_record_list)
                
                # Format check: ensure missing structural variables perfectly map as NaN/None string objects
                # matching your target layout string format requirements:
                # "college_name | city | district | address | dept | intake | intake_year"
                # If fields like city or dept fail to catch values, map explicitly to NaN components:
                for target_column in ["city", "district", "address", "dept", "intake"]:
                    if target_column not in final_extracted_df.columns:
                        final_extracted_df[target_column] = None
                
                # Explicitly align the extraction parameters order
                final_extracted_df = final_extracted_df[[
                    "college_name", "city", "district", "address", "dept", "intake", "intake_year"
                ]]
                
                # Replace None objects visually with pandas NaN handles where required to output identical target structure
                processed_visualization_df = final_extracted_df.copy()
                
                # Persist directly down into local SQLite ecosystem tables
                save_dataframe_to_sqlite(final_extracted_df)
                
                # Synchronize data changes across active application memory registries globally
                st.session_state["shared_dataframe"] = final_extracted_df
                
                st.success(f"🎉 Pipeline execution success! Synchronized {len(final_extracted_df)} records directly into 'matrix_records.db' relational tables.")
                
                st.markdown("### 📋 Final Extracted Structural Database Matrix Output View")
                st.dataframe(processed_visualization_df, use_container_width=True)
                
                # Provide on-the-fly CSV configuration copy downloads
                csv_download_payload = final_extracted_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Processed Matrix as CSV Registered Dataset",
                    data=csv_download_payload,
                    file_name=f"seat_matrix_extracted_{target_year}.csv",
                    mime="text/csv"
                )
            else:
                st.error("⚠️ Document analysis run completed, but the layout extraction worker failed to map matching elements to the structure.")
                
            # Discard local transient operating system file footprints cleanly
            os.unlink(tmp_path)
            
        except Exception as system_error:
            st.error(f"💥 Processing Engine Core Error Interruption: {str(system_error)}")
            st.info("Check if your Groq API token parameter values are authentic and your document structural text layouts fit expected patterns.")
