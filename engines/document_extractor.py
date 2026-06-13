import io
import pandas as pd
from pypdf import PdfReader
from docx import Document

class PragyanDocumentExtractor:
    """
    A unified document abstraction layer for the PragyanAI pipeline.
    Ingests binary streams from PDF, Word, Excel, and CSV files, then normalizes 
    and flattens them into plaintext layouts structured for the downstream parser.
    """
    
    @staticmethod
    def extract_to_text_stream(file_bytes: bytes, file_name: str) -> str:
        """
        Decouples incoming file uploads based on their extension signatures
        and routes them to format-specific plaintext extractors.
        
        Parameters:
            file_bytes (bytes): Raw binary data from the uploaded file.
            file_name (str): The name of the file used to extract the extension type.
            
        Returns:
            str: A unified, newline-separated text stream.
        """
        if not file_name or "." not in file_name:
            raise ValueError("Invalid file structure: Missing explicit extension name signature.")
            
        ext = file_name.split(".")[-1].lower()
        
        if ext == "pdf":
            return PragyanDocumentExtractor._extract_pdf(file_bytes)
        elif ext in ["xlsx", "xls"]:
            return PragyanDocumentExtractor._extract_spreadsheet(file_bytes)
        elif ext == "csv":
            return PragyanDocumentExtractor._extract_csv(file_bytes)
        elif ext in ["docx", "doc"]:
            return PragyanDocumentExtractor._extract_docx(file_bytes)
        else:
            raise ValueError(f"Pipeline Ingestion Aborted: Unsupported file format extension: .{ext}")

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        """
        Reads PDF objects page by page, aggregates localized text layers,
        and strings them together into sequential text boundaries.
        """
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text_accum = []
            
            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    # Injecting a synthetic page boundary trace for easier debugging
                    text_accum.append(f"\n--- PAGE {page_idx + 1} ---\n{page_text}")
                    
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Error parsing PDF layout binaries: {str(e)}")

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        """
        Extracts textual layouts from Microsoft Word files.
        Processes standard body text segments and iterates through rows
        inside any embedded native layout tables.
        """
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_accum = []
            
            # 1. Process standard layout text paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_accum.append(paragraph.text.strip())
                    
            # 2. Extract and format grid alignments inside native DOCX tables
            for table in doc.tables:
                for row in table.rows:
                    # Join individual table columns using clean string pipeline symbols
                    row_data = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_data:
                        text_accum.append(row_data)
                        
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Error extracting structural elements from DOCX: {str(e)}")

    @staticmethod
    def _extract_spreadsheet(file_bytes: bytes) -> str:
        """
        Parses multi-sheet Excel workbooks. Loops through all visible sheets, 
        maps row frames into structured layout objects, and exports them 
        using space-separated layouts.
        """
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            text_accum = []
            
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                
                # Drop rows where every single index column contains NaN data values
                df = df.dropna(how='all')
                
                if not df.empty:
                    # Convert to string format while keeping space alignments intact
                    serialized_sheet = df.to_csv(index=False, sep=" ")
                    text_accum.append(f"\n--- SHEET: {sheet_name} ---\n{serialized_sheet}")
                    
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Error normalizing Excel spreadsheet frames: {str(e)}")

    @staticmethod
    def _extract_csv(file_bytes: bytes) -> str:
        """
        Directly reads unstructured comma-separated variables, isolates data blocks,
        and sanitizes cell properties into standardized text frames.
        """
        try:
            # Using basic automatic fallback engines to prevent encoding violations
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', on_bad_lines='skip')
            df = df.dropna(how='all')
            return df.to_csv(index=False, sep=" ")
        except Exception as e:
            raise RuntimeError(f"Error streaming direct CSV data lines: {str(e)}")
