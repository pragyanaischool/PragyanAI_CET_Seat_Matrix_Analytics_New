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
# 📋 STRUCTURED PYDANTIC SCHEMAS (ALIGNED WITH YOUR GOOGLE COLAB DESIGN)
# ==============================================================================
class SeatMatrixRecord(BaseModel):
    """
    Represents a single course intake row extracted from the seat matrix PDF.
    """
    college_name: str = Field(
        ...,
        description="The name of the institution or university (e.g., 'THE KISHKINDA UNIVERSITY')."
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
# 🚀 PURE GROQ MULTI-LLM CASCADING FAILOVER ORCHESTRATOR
# ==============================================================================
class PragyanEnsembleParser:
    """
    Pure Groq Multi-LLM Cascading Failover Routing Engine for PragyanAI.
    Sequentially transitions down an alternate pool of pure Groq models 
    the millisecond a rate-limit exhaustion exception (429 / Quota) is caught.
    """
    def __init__(self):
        # 🎯 Pure Groq Extraction Model Cascade Pool Configuration Fixed
        self.cascade_pool = [
            {"name": "llama-3.1-8b-instant"},
            {"name": "llama-3.3-70b-versatile"},
            {"name": "llama3-70b-8192"},
            {"name": "llama3-8b-8192"}
        ]
        
        # System instructions mapped directly from your Colab workspace design
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

    def _get_provider_client(self, model_meta: dict):
        """Dynamic runtime adapter tracking and instantiation mapping using pure Groq instances."""
        model_name = model_meta["name"]
        llm = ChatGroq(model_name=model_name, temperature=0.0, max_tokens=4096)
        return llm.with_structured_output(SeatMatrixExtraction)

    def _invoke_cascade_broker(self, chunk: str) -> Optional[SeatMatrixExtraction]:
        """Loops through Groq models dynamically, hot-swapping endpoints upon hitting a 429 error."""
        formatted_messages = self.prompt_template.format_messages(text_chunk=chunk)
        
        # Corrected variable reference loop to prevent runtime reference failures
        for idx, model_meta in enumerate(self.cascade_pool):
            try:
                # Instantiate structured provider adapter dynamically
                structured_extractor = self._get_provider_client(model_meta)
                result = structured_extractor.invoke(formatted_messages)
                
                if result and isinstance(result, SeatMatrixExtraction):
                    return result
                    
            except Exception as e:
                err_msg = str(e)
                # Intercept rate limit, token quota exhaustion, or temporary backend timeouts
                if any(k in err_msg or k in err_msg.lower() for k in ["429", "rate_limit", "limit reached", "quota", "overloaded"]):
                    print(f"    [!] Groq Quota Exhausted for [{model_meta['name']}]. Cascading down to pool index {idx + 1}...")
                    continue
                else:
                    print(f"    [!] Critical Exception hit on Groq layer [{model_meta['name']}]: {err_msg}")
                    raise e
                    
        # Emergency Safe Guard: If all models inside the Groq cluster pool trigger 429s, apply short jitter backoff
        print("    [⚠️] CRITICAL: Entire Groq failover cluster exhausted. Injecting 5s emergency jitter sleep...")
        time.sleep(5.0 + random.uniform(0.5, 1.5))
        
        # Final emergency retry run using the robust versatility model tier
        emergency_extractor = self._get_provider_client({"name": "llama-3.3-70b-versatile"})
        return emergency_extractor.invoke(formatted_messages)

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        """Processes raw document bytes through text chunk loops utilizing the pure Groq cascade system."""
        # Open up isolated localized secure disk paths to pipe raw input streams securely
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            absolute_path = os.path.abspath(temp_pdf.name)
            
        markdown_text = ""
        try:
            if IBMDoclingConverter is not None:
                markdown_text = IBMDoclingConverter().convert(absolute_path).document.export_to_markdown()
        except Exception as docling_err:
            print(f"[Docling Structural Parsing Error] Layout conversion failed: {str(docling_err)}")
        finally:
            if os.path.exists(absolute_path):
                os.remove(absolute_path)
                
        if not markdown_text.strip():
            return pd.DataFrame()

        # Fragment markdown tables safely matching Colab optimization metrics to respect token dimensions
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500, separators=["\n\n", "\n", " "])
        chunks = text_splitter.split_text(markdown_text)
        
        compiled_records = []
        for idx, chunk in enumerate(chunks):
            # Performance Optimization Step: Skip headers or footers without tabular metrics
            if "COURSE NAME" not in chunk.upper() and "INTAKE" not in chunk.upper() and len(chunk.strip()) < 120:
                continue
                
            try:
                extraction_result = self._invoke_cascade_broker(chunk)
                if extraction_result and extraction_result.records:
                    # Unpack structured data entities straight into row collectors
                    for record in extraction_result.records:
                        record_dict = record.model_dump()
                        record_dict['intake_year'] = int(intake_year)
                        compiled_records.append(record_dict)
            except Exception as chunk_err:
                print(f"    [!] Dropped text chunk slice step index {idx+1} due to extraction failure: {str(chunk_err)}")
                continue

        if not compiled_records:
            return pd.DataFrame()
            
        return pd.DataFrame(compiled_records)
        
