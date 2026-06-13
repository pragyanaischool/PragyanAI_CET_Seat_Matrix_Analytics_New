import os
import re
import json
import tempfile
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# ==============================================================================
# 🎯 COMPREHENSIVE AI PARSING ENGINE CORES - DYNAMIC IMPORT PROTECTION
# ==============================================================================
try:
    from docling.document_converter import DocumentConverter as IBMDoclingConverter
except ImportError:
    IBMDoclingConverter = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None


class PragyanEnsembleParser:
    """
    Multi-Engine Layout Consensus Parsing Core Engine for PragyanAI.
    Combines text and markdown layers across multiple document extraction frameworks, 
    then applies Llama 3.3 to guarantee clean row multiplication per discipline.
    """
    def __init__(self):
        # High-capacity versatile model line configured for zero-shot tabular construction
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        
        # Explicit consensus template designed to force extraction out of layered layout blocks
        self.analysis_template = PromptTemplate.from_template("""
        You are an elite data engineering extraction agent specialized in structural matrix normalization.
        Analyze the following comprehensive text streams collected from multiple PDF extraction engines.
        
        The input data contains engineering seat allocation matrices. Even if multiple course/department names 
        are lumped together horizontally within Markdown cells (|) or visually wrapped across lines, you must split them apart.
        
        For EVERY single individual engineering branch found (e.g., Computer Science, Electrical, Mechanical, Civil, Automobile, etc.), 
        generate a separate JSON object. Keep the college_name, city, district, and address completely identical for each row.
        
        Return the results strictly formatted inside a single valid JSON array of objects. 
        Do not include any preamble, markdown code blocks, or conversational commentary.
        
        Schema Format to Enforce:
        [
          {{
            "college_name": "Government Engineering College",
            "city": "Challakere",
            "district": "Chitradurga",
            "address": "BALLARI ROAD CHALLAKERE,CHITRADURGA",
            "dept": "COMPUTER SCIENCE AND ENGINEERING",
            "intake": 60
          }},
          {{
            "college_name": "Government Engineering College",
            "city": "Challakere",
            "district": "Chitradurga",
            "address": "BALLARI ROAD CHALLAKERE,CHITRADURGA",
            "dept": "AUTOMOBILE ENGINEERING",
            "intake": 60
          }}
        ]
        
        Ensemble Raw Text Stream Payload to Process:
        {combined_text_stream}
        """)

    def extract_all_engines(self, file_bytes: bytes) -> str:
        """Runs Docling, pdfplumber, and PyMuPDF concurrently to assemble a high-density layout string."""
        combined_outputs = []
        
        # Create a secure absolute path temporary file block to prevent background disk drop trace failures
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            absolute_path = os.path.abspath(temp_pdf.name)
            
        try:
            # --- ENGINE REGION A: IBM Docling (Premium Structured Markdown Grid Tables) ---
            if IBMDoclingConverter is not None:
                try:
                    converter = IBMDoclingConverter()
                    docling_doc = converter.convert(absolute_path)
                    markdown_text = docling_doc.document.export_to_markdown()
                    if markdown_text and markdown_text.strip():
                        combined_outputs.append(f"=== IBM DOCLING MARKDOWN EXTRACT ===\n{markdown_text}")
                except Exception as e:
                    print(f"[Ensemble Node Alert] IBM Docling processing window bypassed: {str(e)}")

            # --- ENGINE REGION B: pdfplumber (Preserved Spacing Visual Character Arrays) ---
            if pdfplumber is not None:
                try:
                    text_accum = []
                    with pdfplumber.open(absolute_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text(layout=True)
                            if page_text:
                                text_accum.append(page_text)
                    if text_accum:
                        combined_outputs.append(f"=== PDFPLUMBER VISUAL LAYOUT EXTRACT ===\n" + "\n".join(text_accum))
                except Exception as e:
                    print(f"[Ensemble Node Alert] pdfplumber character extraction bypassed: {str(e)}")

            # --- ENGINE REGION C: PyMuPDF / fitz (Bounding Box Layout Block Ingestion) ---
            if fitz is not None:
                try:
                    text_accum = []
                    doc = fitz.open(absolute_path)
                    for page in doc:
                        blocks = page.get_text("blocks")
                        for b in blocks:
                            text_accum.append(b[4]) # Pull string character content safely out of block tuple fields
                    if text_accum:
                        combined_outputs.append(f"=== PYMUPDF SEPARATE BLOCK EXTRACT ===\n" + "\n".join(text_accum))
                except Exception as e:
                    print(f"[Ensemble Node Alert] PyMuPDF semantic bounding blocks bypassed: {str(e)}")
                    
        finally:
            # Secure Housekeeping Loop: Always free the file resource descriptor from host paths
            if os.path.exists(absolute_path):
                try:
                    os.remove(absolute_path)
                except Exception:
                    pass
                
        if not combined_outputs:
            raise RuntimeError("Orchestration Framework Fault: All three underlying extraction engine adapters failed simultaneously.")
            
        return "\n\n".join(combined_outputs)

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        """
        Processes combined multiline layout logs through consensus-driven prompt metrics
        to output clean, branch-multiplied structural DataFrames.
        """
        # 1. Harvest layout outputs across all working libraries
        text_stream = self.extract_all_engines(file_bytes)
        
        # 2. Compile structural payloads through Groq framework pipelines
        try:
            chain = self.analysis_template | self.llm
            response = chain.invoke({"combined_text_stream": text_stream}).content
            clean_content = response.strip()
            
            # Isolate pure JSON array tokens, stripping out markdown formatting backticks safely
            if "```json" in clean_content:
                json_str = clean_content.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_content:
                json_str = clean_content.split("```")[-1].split("```")[0].strip()
            else:
                start_idx = clean_content.find("[")
                end_idx = clean_content.rfind("]") + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = clean_content[start_idx:end_idx]
                else:
                    json_str = clean_content
                
            parsed_json = json.loads(json_str)
            df = pd.DataFrame(parsed_json)
            
            if not df.empty:
                # Post-Extraction Ingest Sanitization Loops
                df['intake_year'] = int(intake_year)
                df = df.dropna(subset=['dept'])
                
                df['college_name'] = df['college_name'].astype(str).str.strip()
                df['city'] = df['city'].astype(str).str.strip()
                df['district'] = df['district'].astype(str).str.strip()
                df['dept'] = df['dept'].astype(str).str.strip().str.upper()
                
                # Drop exact structural copies while completely protecting duplicate branches across different institutions
                df = df.drop_duplicates(
                    subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                    keep='first'
                )
                return df.reset_index(drop=True)
                
        except Exception as e:
            print(f"[Critical Orchestration LLM Core Exception Encountered]: {str(e)}")
            
        return pd.DataFrame()
