import os
import re
import json
import tempfile
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# --- Fail-Safe Structural Import Checking Boundaries ---
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
    Ensemble Layout Consensus Parsing Core Engine for PragyanAI.
    Combines text and markdown layers across multiple document extraction frameworks, 
    then applies Llama 3.3 to guarantee clean row multiplication per discipline.
    """
    def __init__(self):
        # Initializing high-throughput versatile model to handle formatting recovery
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        
        self.consensus_template = PromptTemplate.from_template("""
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
        """Extracts text structures from Docling, pdfplumber, and PyMuPDF concurrently."""
        combined_outputs = []
        
        # Isolate layout files using absolute path temporary generation
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            absolute_path = os.path.abspath(temp_pdf.name)
            
        try:
            # 1. Strategy A: IBM Docling (Structural Markdown Tables)
            if IBMDoclingConverter is not None:
                try:
                    converter = IBMDoclingConverter()
                    docling_doc = converter.convert(absolute_path)
                    markdown_text = docling_doc.document.export_to_markdown()
                    if markdown_text.strip():
                        combined_outputs.append(f"=== IBM DOCLING MARKDOWN EXTRACT ===\n{markdown_text}")
                except Exception as e:
                    print(f"[Ensemble Guard] Docling processing skipped: {str(e)}")

            # 2. Strategy B: pdfplumber (Preserved Spacing Visual Grid Strings)
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
                    print(f"[Ensemble Guard] pdfplumber processing skipped: {str(e)}")

            # 3. Strategy C: PyMuPDF / fitz (Block Segment Boundary Parsing)
            if fitz is not None:
                try:
                    text_accum = []
                    doc = fitz.open(absolute_path)
                    for page in doc:
                        blocks = page.get_text("blocks")
                        for b in blocks:
                            text_accum.append(b[4])
                    if text_accum:
                        combined_outputs.append(f"=== PYMUPDF SEPARATE BLOCK EXTRACT ===\n" + "\n".join(text_accum))
                except Exception as e:
                    print(f"[Ensemble Guard] PyMuPDF processing skipped: {str(e)}")
                    
        finally:
            if os.path.exists(absolute_path):
                os.remove(absolute_path)
                
        if not combined_outputs:
            raise RuntimeError("Orchestration Fault: All structural text extraction frameworks failed.")
            
        return "\n\n".join(combined_outputs)

    def analyze_and_extract_matrix(self, file_bytes: bytes, intake_year: int) -> pd.DataFrame:
        """Processes combined multiline layouts via consensus to generate row-multiplied outputs."""
        text_stream = self.extract_all_engines(file_bytes)
        
        try:
            chain = self.consensus_template | self.llm
            response = chain.invoke({"combined_text_stream": text_stream}).content
            clean_content = response.strip()
            
            # Isolate JSON array markers cleanly
            if "```json" in clean_content:
                json_str = clean_content.split("```json")[-1].split("```")[0].strip()
            elif "```" in clean_content:
                json_str = clean_content.split("```")[-1].split("```")[0].strip()
            else:
                start_idx = clean_content.find("[")
                end_idx = clean_content.rfind("]") + 1
                json_str = clean_content[start_idx:end_idx] if start_idx != -1 else clean_content
                
            parsed_json = json.loads(json_str)
            df = pd.DataFrame(parsed_json)
            
            if not df.empty:
                df['intake_year'] = int(intake_year)
                df = df.dropna(subset=['dept'])
                
                # Structural normalization pipelines
                df['college_name'] = df['college_name'].astype(str).str.strip()
                df['city'] = df['city'].astype(str).str.strip()
                df['district'] = df['district'].astype(str).str.strip()
                df['dept'] = df['dept'].astype(str).str.strip().str.upper()
                
                # Drop structural duplicates based on core metrics criteria
                df = df.drop_duplicates(
                    subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                    keep='first'
                )
                return df.reset_index(drop=True)
                
        except Exception as e:
            print(f"[Critical Ensemble Exception Encountered]: {str(e)}")
            
        return pd.DataFrame()
