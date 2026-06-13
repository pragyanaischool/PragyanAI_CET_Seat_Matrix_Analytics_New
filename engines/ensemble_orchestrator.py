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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from docling.document_converter import DocumentConverter as IBMDoclingConverter
except ImportError:
    IBMDoclingConverter = None

# ==============================================================================
# 📋 STRUCTURED PYDANTIC SCHEMAS (EXTRACTED FROM YOUR JUPYTER WORKSPACE)
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
# 🚀 MULTI-LLM CASCADING FAILOVER ORCHESTRATOR
# ==============================================================================
class PragyanEnsembleParser:
    """
    Multi-Provider Multi-LLM Cascading Failover Routing Engine for PragyanAI.
    Sequentially hops through your specified model pool the millisecond a 
    rate-limit exhaustion exception (429 / Quota Exceeded) is caught.
    """
    def __init__(self):
        # Master Sequence Failover Chain Pool
        self.cascade_pool = [
            {"provider": "groq", "name": "llama-3.1-8b-instant"},
            {"provider": "groq", "name": "llama-3.3-70b-versatile"},
            {"provider": "openai-oss", "name": "openai/gpt-oss-120b"},
            {"provider": "openai-oss", "name": "openai/gpt-oss-20b"},
            {"provider": "qwen", "name": "qwen/qwen3-32b"}
        ]
        
        # System instructions mapped directly from your Colab workspace
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
        provider = model_meta["provider"]
        model_name = model_meta["name"]
        
        if provider == "groq":
            llm = ChatGroq(model_name=model_name, temperature=0.0, max_tokens=4096)
            return llm.with_structured_output(SeatMatrixExtraction)
            
        elif provider in ["openai-oss", "qwen"]:
            custom_base_url = os.getenv("OPEN_OSS_BASE_URL", "https://api.openai.com/v1")
            custom_api_key = os.getenv("OPEN_OSS_API_KEY", os.getenv("GROQ_API_KEY"))
            
            llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.0,
                max_tokens=4096,
                openai_api_base=custom_base_url,
                openai_api_key=custom_api_key
            )
            return llm.with_structured_output(SeatMatrixExtraction)
        else:
            raise ValueError(f"Unknown system interface provider context specified: {provider}")

    def _invoke_cascade_broker(self, chunk: str) -> Optional[SeatMatrixExtraction]:
        formatted_messages = self.prompt_template.format_messages(text_chunk=chunk)
        
        for idx, model_meta in enumerate(self.cascade_pool):
            try:
                structured_extractor = self._get_provider_client(model_meta)
                result = structured_extractor.invoke(formatted_messages)
                if result and isinstance(result, SeatMatrixExtraction):
                    return result
            except Exception as e:
                err_msg = str(e)
                if any(k in err_msg or k in err_msg.lower() for k in ["429", "rate_limit", "limit reached", "quota", "overloaded"]):
                    print(f"    [!] Quota Exhausted for [{model_meta['name']}]. Cascading down to pool index {idx + 1}...")
                    continue
                else:
                    raise e
                    
        print("    [⚠️] CRITICAL: Entire LLM failover cluster exhausted. Injecting emergency jitter sleep...")
        time.sleep(5.0 + random.uniform(0.5, 1.5))
        
        emergency_extractor = self._get_provider_client({"provider": "groq", "name": "llama-3.3-70b-versatile"})
        return emergency_extractor.invoke(formatted_messages)

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            absolute_path = os.path.abspath(temp_pdf.name)
            
        markdown_text = ""
        try:
            if IBMDoclingConverter is not None:
                markdown_text = IBMDoclingConverter().convert(absolute_path).document.export_to_markdown()
        except Exception as docling_err:
            print(f"[Docling Error] Layout conversion failed: {str(docling_err)}")
        finally:
            if os.path.exists(absolute_path):
                os.remove(absolute_path)
                
        if not markdown_text.strip():
            return pd.DataFrame()

        # Chunk configuration matched to your notebook dimensions
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=500, separators=["\n\n", "\n", " "])
        chunks = text_splitter.split_text(markdown_text)
        
        compiled_records = []
        for idx, chunk in enumerate(chunks):
            if "COURSE NAME" not in chunk.upper() and "INTAKE" not in chunk.upper() and len(chunk.strip()) < 120:
                continue
                
            try:
                extraction_result = self._invoke_cascade_broker(chunk)
                if extraction_result and extraction_result.records:
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
        
