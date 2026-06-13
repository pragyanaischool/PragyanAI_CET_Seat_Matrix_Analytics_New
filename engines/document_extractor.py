import io
import os
import tempfile
import pandas as pd
from docx import Document

# --- AI Extraction Engines Import Guard Boundaries ---
try:
    from docling.document_converter import DocumentConverter as IBMDoclingConverter
except ImportError:
    IBMDoclingConverter = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class PragyanDocumentExtractor:
    """
    Enterprise ensemble data extraction and normalization engine for PragyanAI.
    Leverages IBM's Docling to transform highly unpredictable administrative PDFs 
    into structured, layout-healed markdown tables, with a resilient pdfplumber fallback.
    """
    
    @staticmethod
    def extract_to_text_stream(file_bytes: bytes, file_name: str) -> str:
        """
        Routes incoming binary blocks to specific data parsing handlers based on file type.
        
        Parameters:
            file_bytes (bytes): The raw binary content of the uploaded matrix file.
            file_name (str): The original filename used to resolve format extensions.
            
        Returns:
            str: Clean, layout-preserved text or markdown matrix stream.
        """
        if not file_name or "." not in file_name:
            raise ValueError("Invalid document reference: Missing explicit format extension suffix.")
            
        ext = file_name.split(".")[-1].lower()
        
        if ext == "pdf":
            return PragyanDocumentExtractor._extract_pdf_pipeline(file_bytes)
        elif ext in ["xlsx", "xls"]:
            return PragyanDocumentExtractor._extract_spreadsheet(file_bytes)
        elif ext == "csv":
            return PragyanDocumentExtractor._extract_csv(file_bytes)
        elif ext in ["docx", "doc"]:
            return PragyanDocumentExtractor._extract_docx(file_bytes)
        else:
            raise ValueError(f"Ingestion Core Aborted: Unsupported file format specification: .{ext}")

    @staticmethod
    def _extract_pdf_pipeline(file_bytes: bytes) -> str:
        """
        Executes Docling for superior table structure and grid markdown extraction.
        Uses a secure absolute path temporary file block to eliminate cross-thread path loss.
        """
        # Lock down a unique temporary footprint path on disk
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            absolute_temp_path = os.path.abspath(temp_pdf.name)

        try:
            # --- STRATEGY A: IBM DOCLING (Premium Grid & Markdown Engine) ---
            if IBMDoclingConverter is not None:
                try:
                    converter = IBMDoclingConverter()
                    docling_doc = converter.convert(absolute_temp_path)
                    markdown_output = docling_doc.document.export_to_markdown()
                    
                    if markdown_output and markdown_output.strip():
                        return markdown_output
                except Exception as e:
                    print(f"[Extractor Alert] IBM Docling parsing phase bypassed, cascading to fallback: {str(e)}")

            # --- STRATEGY B: SPATIAL LOCAL ENFORCER (pdfplumber layout backup) ---
            if pdfplumber is not None:
                print("[Extractor Warning] Running pdfplumber layout fallback...")
                text_accum = []
                with pdfplumber.open(absolute_temp_path) as pdf:
                    for idx, page in enumerate(pdf.pages):
                        page_text = page.extract_text(layout=True)
                        if page_text:
                            text_accum.append(f"\n--- PAGE {idx + 1} ---\n{page_text}")
                return "\n".join(text_accum)
            else:
                raise RuntimeError("Critical System Fault: No active document parser extensions found inside local runtime environment.")
                
        finally:
            # Secure Housekeeping Loop: Always remove the temporary artifact using the absolute identifier
            if os.path.exists(absolute_temp_path):
                try:
                    os.remove(absolute_temp_path)
                except Exception:
                    pass

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        """Flattens paragraphs and multi-column tables cleanly from Word documents."""
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_accum = []
            
            # Extract standard linear paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_accum.append(paragraph.text.strip())
                    
            # Extract text arrays from nested table layouts
            for table in doc.tables:
                for row in table.rows:
                    row_data = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_data:
                        text_accum.append(row_data)
                        
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Advanced DOCX Extraction Module Exception encountered: {str(e)}")

    @staticmethod
    def _extract_spreadsheet(file_bytes: bytes) -> str:
        """Transforms multi-sheet Excel records into space-delimited text frames."""
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            text_accum = []
            
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name).dropna(how='all')
                if not df.empty:
                    serialized_sheet = df.to_csv(index=False, sep=" ")
                    text_accum.append(f"\n--- SHEET: {sheet_name} ---\n{serialized_sheet}")
                    
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Advanced Excel Spreadsheet compilation sequence error: {str(e)}")

    @staticmethod
    def _extract_csv(file_bytes: bytes) -> str:
        """Normalizes standard raw CSV data records into clean space-separated lines."""
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', on_bad_lines='skip').dropna(how='all')
            return df.to_csv(index=False, sep=" ")
        except Exception as e:
            raise RuntimeError(f"CSV payload processing stream exception hit: {str(e)}")
