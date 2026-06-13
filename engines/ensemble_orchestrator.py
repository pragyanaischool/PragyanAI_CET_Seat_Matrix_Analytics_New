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
# 📋 STRUCTURED PYDANTIC SCHEMAS (ALIGNED WITH COLAB FRAMEWORKS)
# ==============================================================================
class SeatMatrixRecord(BaseModel):
    college_name: str = Field(..., description="Name of the educational institution (e.g., 'THE KISHKINDA UNIVERSITY')")
    city: Optional[str] = Field(None, description="The specific city node where the campus is located.")
    district: Optional[str] = Field(None, description="The localized administrative district name.")
    address: Optional[str] = Field(None, description="The verified full postal or location address.")
    dept: Optional[str] = Field(None, description="The engineering specialization/branch label (e.g., 'COMPUTER SCIENCE AND ENGINEERING').")
    intake: Optional[int] = Field(None, description="The total seat intake number allocated for this option line entry.")
    intake_year: Optional[int] = Field(2024, description="The tracking academic horizon year identifier.")

class SeatMatrixExtraction(BaseModel):
    records: List[SeatMatrixRecord] = Field(default=[], description="Structured collection block holding parsed matrix rows.")

# ==============================================================================
# 🚀 ADVANCED CASCADING FAILOVER ENSEMBLE ORCHESTRATOR
# ==============================================================================
class PragyanEnsembleParser:
    """
    Multi-Provider Multi-LLM Cascading Failover Routing Engine for PragyanAI.
    Sequentially hops through user-specified custom model architectures and Groq tiers
    the millisecond a rate-limit exhaustion exception (429 / Quota) is caught.
    """
    def __init__(self):
        # 🎯 Master Orchestration Cascade Chain Pool Configuration
        self.cascade_pool = [
            {"provider": "groq", "name": "llama-3.1-8b-instant"},
            {"provider": "groq", "name": "llama-3.3-70b-versatile"},
            {"provider": "openai-oss", "name": "openai/gpt-oss-120b"},
            {"provider": "openai-oss", "name": "openai/gpt-oss-20b"},
            {"provider": "qwen", "name": "qwen/qwen3-32b"}
        ]
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite data extraction assistant specialized in institutional seat matrix optimization.\n"
                "Review the raw markdown layout text chunk carefully and isolate all tabular data rows.\n"
                "Ensure institutional metadata properties (college_name, city, district, address) are explicitly duplicated "
                "and populated for each individual engineering branch found."
            )),
            ("user", "Here is a section of the parsed seat matrix document:\n\n{text_chunk}\n\nExtract all records matching the strict JSON format parameters.")
        ])

    def _get_provider_client(self, model_meta: dict):
        """Dynamic runtime adapter tracking and instantiation mapping."""
        provider = model_meta["provider"]
        model_name = model_meta["name"]
        
        if provider == "groq":
            llm = ChatGroq(model_name=model_name, temperature=0.0, max_tokens=4096)
            return llm.with_structured_output(SeatMatrixExtraction)
            
        elif provider in ["openai-oss", "qwen"]:
            # Pull localized open source endpoints securely from custom configurations or environments
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
        """Loops through models dynamically, hot-swapping endpoints upon hitting a 429 error."""
        formatted_messages = self.prompt_template.format_messages(text_chunk=chunk)
        
        for idx, model_meta in enumerate(self.cascade_pool):
            try:
                # Instantiate structured provider adapter dynamically
                structured_extractor = self._get_provider_client(model_meta)
                result = structured_extractor.invoke(formatted_messages)
                
                if result and isinstance(result, SeatMatrixExtraction):
                    return result
                    
            except Exception as e:
                err_msg = str(e)
                # Intercept rate limit, quota exhaustion, or overloaded hosting boundaries
                if any(k in err_msg or k in err_msg.lower() for k in ["429", "rate_limit", "limit reached", "quota", "overloaded"]):
                    print(f"    [!] Quota Exhausted for [{model_meta['name']}]. Cascading down to pool index {idx + 1}...")
                    continue
                else:
                    print(f"    [!] Critical Exception hit on current model layer [{model_meta['name']}]: {err_msg}")
                    raise e
                    
        # Emergency Safe Guard: If all models inside the cluster pool trigger 429s, apply short jitter backoff
        print("    [⚠️] CRITICAL: Entire LLM failover cluster exhausted. Injecting 5s emergency jitter sleep...")
        time.sleep(5.0 + random.uniform(0.5, 1.5))
        
        emergency_extractor = self._get_provider_client({"provider": "groq", "name": "llama-3.3-70b-versatile"})
        return emergency_extractor.invoke(formatted_messages)

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        """Processes raw document bytes through text chunk loops utilizing the multi-LLM router canvas."""
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=400, separators=["\n\n", "\n", " "])
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
