import os
import re
import json
import time
import random
import tempfile
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from docling.document_converter import DocumentConverter as IBMDoclingConverter
except ImportError:
    IBMDoclingConverter = None

# ==============================================================================
# 📋 1. DEFINE TARGET DATA STRUCTURE (Pydantic - Preserved from Workspace)
# ==============================================================================
class SeatMatrixRecord(BaseModel):
    """
    Represents a single course intake row extracted from the seat matrix PDF.
    """
    college_name: str = Field(
        ..., 
        description="The name of the institution or university (e.g., 'THE KISHKINDA UNIVERSITY')"
    )
    city: Optional[str] = Field(
        None, 
        description="The city where the college is located if identifiable from the text or address."
    )
    district: Optional[str] = Field(
        None, 
        description="The district where the college is located."
    )
    address: Optional[str] = Field(
        None, 
        description="The full address of the college/university."
    )
    dept: Optional[str] = Field(
        None, 
        description="The full name of the department/course (e.g., 'COMPUTER SCIENCE AND ENGG')."
    )
    intake: Optional[int] = Field(
        None, 
        description="The numeric total intake or KEA seat value assigned for this course."
    )
    intake_year: Optional[int] = Field(
        2024, 
        description="The intake year. Default to 2024 unless the PDF text explicitly specifies another year."
    )

class SeatMatrixExtraction(BaseModel):
    """
    Container to hold list of structured records returned from the LLM.
    """
    records: List[SeatMatrixRecord] = Field(
        default=[], 
        description="A list of structured seat matrix data records extracted from the parsed document chunk."
    )


# ==============================================================================
# 🚀 2. THE PIPELINE ORCHESTRATOR CLASS (With Adaptive Cascade Failover)
# ==============================================================================
class PragyanEnsembleParser:
    """
    High-Throughput Pure Groq Multi-LLM Cascading Failover Ingestion Engine.
    Combines your original PDFToCSVPipeline structural logic with multi-tier backoff 
    protection to handle high-volume text chunk loads seamlessly without API drops.
    """
    def __init__(self, groq_api_key: Optional[str] = None):
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        
        # 🎯 Dynamic Multi-Tier Groq Cascade Model Pool 
        self.cascade_pool = [
            {"name": "llama-3.3-70b-versatile"},
            {"name": "llama-3.1-8b-instant"},
            {"name": "llama3-70b-8192"},
            {"name": "llama3-8b-8192"}
        ]
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert data extraction assistant. Your task is to extract structured tabular information "
                "from college seat matrix documents.\n\n"
                "Review the provided raw markdown carefully. Pay special attention to: \n"
                "- Institutional details (Name, Address, City, District) which are usually listed before a sequence of tables.\n"
                "- Course or Department entries and their associated total intake values.\n"
                "- If a field (like city, district, or address) is missing, infer it from the context if possible, "
                "or leave it empty/null if unknown."
            )),
            ("user", "Here is a section of the parsed seat matrix document:\n\n{text_chunk}\n\nExtract all records.")
        ])

    def _get_structured_extractor(self, model_name: str):
        llm = ChatGroq(
            model=model_name,
            temperature=0,  
            api_key=self.api_key,
            max_tokens=4096
        )
        return llm.with_structured_output(SeatMatrixExtraction)

    def _invoke_cascade_broker(self, chunk: str) -> Optional[SeatMatrixExtraction]:
        """
        Loops through the multi-LLM model pool with exponential backoff and random jitter 
        to survive rate limits (429) during massive extraction pipelines.
        """
        formatted_messages = self.prompt_template.format_messages(text_chunk=chunk)
        base_delay = 2.0  
        max_retries = 4
        
        for idx, model_meta in enumerate(self.cascade_pool):
            retries = 0
            model_name = model_meta["name"]
            
            while retries < max_retries:
                try:
                    structured_extractor = self._get_structured_extractor(model_name)
                    result = structured_extractor.invoke(formatted_messages)
                    
                    if result and isinstance(result, SeatMatrixExtraction):
                        return result
                except Exception as e:
                    err_msg = str(e)
                    if any(k in err_msg or k in err_msg.lower() for k in ["429", "rate_limit", "limit reached", "quota", "overloaded"]):
                        retries += 1
                        sleep_duration = (base_delay ** retries) + random.uniform(0.5, 1.5)
                        print(f"    [!] Rate Limit caught for [{model_name}]. Retry {retries}/{max_retries}. Sleeping for {sleep_duration:.2f}s...")
                        time.sleep(sleep_duration)
                        continue
                    else:
                        print(f"    [!] Non-rate exception hit on model layer [{model_name}]: {err_msg}")
                        raise e
            
            print(f"    [!] Model tier [{model_name}] completely exhausted or rate-limited. Cascading down chain...")

        print("    [⚠️] CRITICAL: Entire Groq cascade pool rate-limited. Forcing emergency jitter sleep...")
        time.sleep(8.0 + random.uniform(1.0, 3.0))
        
        emergency_extractor = self._get_structured_extractor("llama-3.3-70b-versatile")
        return emergency_extractor.invoke(formatted_messages)

    def parse_pdf_to_markdown(self, pdf_path_or_bytes) -> str:
        if isinstance(pdf_path_or_bytes, bytes):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(pdf_path_or_bytes)
                absolute_path = os.path.abspath(temp_pdf.name)
            is_temp = True
        else:
            absolute_path = pdf_path_or_bytes
            is_temp = False

        markdown_text = ""
        try:
            print(f"[*] Extracting PDF layout using Docling: {absolute_path}...")
            if IBMDoclingConverter is not None:
                converter = IBMDoclingConverter()
                result = converter.convert(absolute_path)
                markdown_text = result.document.export_to_markdown()
                print("[+] PDF parsed successfully!")
        except Exception as e:
            print(f"[!] Critical Error during Docling conversion layer: {str(e)}")
        finally:
            if is_temp and os.path.exists(absolute_path):
                os.remove(absolute_path)

        return markdown_text

    def extract_structured_data(self, markdown_text: str, intake_year: int = 2024) -> List[SeatMatrixRecord]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000, 
            chunk_overlap=500,
            separators=["\n\n", "\n", " "]
        )
        chunks = text_splitter.split_text(markdown_text)
        print(f"[*] Document split into {len(chunks)} chunks for LLM processing...")

        all_extracted_records = []

        for i, chunk in enumerate(chunks):
            if "COURSE NAME" not in chunk.upper() and "INTAKE" not in chunk.upper() and len(chunk.strip()) < 120:
                continue
                
            print(f"    -> Extracting chunk {i+1}/{len(chunks)}...")
            try:
                extraction_result = self._invoke_cascade_broker(chunk)
                
                if extraction_result and extraction_result.records:
                    print(f"        [✓] Extracted {len(extraction_result.records)} records from this chunk.")
                    for record in extraction_result.records:
                        if not record.intake_year or record.intake_year == 2024:
                            record.intake_year = int(intake_year)
                        all_extracted_records.append(record)
                else:
                    print("        [-] No records identified in this chunk.")
            except Exception as e:
                print(f"        [!] Error extracting chunk {i+1}: {e}")
                continue

        return all_extracted_records

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("Groq API key missing.")
            
        markdown_text = self.parse_pdf_to_markdown(file_bytes)
        if not markdown_text.strip():
            return pd.DataFrame()
            
        records = self.extract_structured_data(markdown_text, intake_year)
        raw_list_of_dicts = [record.model_dump() for record in records]
        df = pd.DataFrame(raw_list_of_dicts)
        
        if not df.empty:
            columns_order = ["college_name", "city", "district", "address", "dept", "intake", "intake_year"]
            df = df.reindex(columns=columns_order)
        return df

    def run(self, pdf_path: str, output_csv_path: str, intake_year: int = 2024):
        if not self.api_key:
            self.api_key = input("Enter Groq API Key: ").strip()

        markdown_text = self.parse_pdf_to_markdown(pdf_path)
        records = self.extract_structured_data(markdown_text, intake_year)
        
        print(f"[*] Exporting parsed data to CSV format...")
        raw_list_of_dicts = [record.model_dump() for record in records]
        df = pd.DataFrame(raw_list_of_dicts)
        
        if not df.empty:
            columns_order = ["college_name", "city", "district", "address", "dept", "intake", "intake_year"]
            df = df.reindex(columns=columns_order)
            df.to_csv(output_csv_path, index=False)
            print(f"[✓] Data pipeline run completed! File saved to: {output_csv_path}")
            
