import io
import pandas as pd
import pdfplumber
from docx import Document

class PragyanDocumentExtractor:
    """
    Unified ingestion utility for PragyanAI's analytics pipeline.
    Streams binary file buffers (PDF, DOCX, XLSX, CSV) into memory and flattens
    them into standardized text dumps while meticulously preserving tabular alignment.
    """
    
    @staticmethod
    def extract_to_text_stream(file_bytes: bytes, file_name: str) -> str:
        """
        Decouples incoming binary uploads based on their extension signatures
        and routes them to format-specific extraction engines.
        
        Parameters:
            file_bytes (bytes): Raw binary payload from the file stream.
            file_name (str): Original string name of the file to slice the extension tracker.
            
        Returns:
            str: Normalized, newline-separated textual matrix context.
        """
        if not file_name or "." not in file_name:
            raise ValueError("Invalid document reference: Missing explicit format extension suffix.")
            
        ext = file_name.split(".")[-1].lower()
        
        if ext == "pdf":
            return PragyanDocumentExtractor._extract_pdf_with_layout(file_bytes)
        elif ext in ["xlsx", "xls"]:
            return PragyanDocumentExtractor._extract_spreadsheet(file_bytes)
        elif ext == "csv":
            return PragyanDocumentExtractor._extract_csv(file_bytes)
        elif ext in ["docx", "doc"]:
            return PragyanDocumentExtractor._extract_docx(file_bytes)
        else:
            raise ValueError(f"Ingestion Core Aborted: Unsupported file format specification: .{ext}")

    @staticmethod
    def _extract_pdf_with_layout(file_bytes: bytes) -> str:
        """
        Extracts textual matrices using pdfplumber's spatial alignment parameters.
        Preserves structural column spaces, resolving layout signature clipping.
        """
        try:
            text_accum = []
            # Open binary array objects completely in-memory using byte-streams
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    # layout=True forces the engine to preserve structural gaps across columns,
                    # mimicking the visual table structure of the original government document
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text_accum.append(f"\n--- PAGE {page_idx + 1} ---\n{page_text}")
                        
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Advanced PDF Layout Extraction Hub failed to parse pages: {str(e)}")

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        """
        Extracts parameters from Microsoft Word documents. 
        Iterates across core body text blocks and embedded tabular grids.
        """
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_accum = []
            
            # 1. Capture standard body paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_accum.append(paragraph.text.strip())
                    
            # 2. Flatten document table cells using distinct divider pipelines (|)
            for table in doc.tables:
                for row in table.rows:
                    row_data = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_data:
                        text_accum.append(row_data)
                        
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Advanced DOCX Extraction Framework encountered error: {str(e)}")

    @staticmethod
    def _extract_spreadsheet(file_bytes: bytes) -> str:
        """
        Iterates over multi-sheet Excel books. Parses row values 
        and outputs space-separated representations.
        """
        try:
            excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
            text_accum = []
            
            for sheet_name in excel_file.sheet_names:
                df = excel_file.parse(sheet_name)
                # Drop structural index lines where every variable contains null data values
                df = df.dropna(how='all')
                
                if not df.empty:
                    # Export cell tracks using a unified single-space delimiter alignment configuration
                    serialized_sheet = df.to_csv(index=False, sep=" ")
                    text_accum.append(f"\n--- SHEET: {sheet_name} ---\n{serialized_sheet}")
                    
            return "\n".join(text_accum)
        except Exception as e:
            raise RuntimeError(f"Workbook Extraction Subsystems failed to serialize sheets: {str(e)}")

    @staticmethod
    def _extract_csv(file_bytes: bytes) -> str:
        """
        Directly reads unformatted comma-separated entries, strips formatting artifacts,
        and translates rows into space-delimited text blocks.
        """
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding='utf-8', on_bad_lines='skip')
            df = df.dropna(how='all')
            return df.to_csv(index=False, sep=" ")
        except Exception as e:
            raise RuntimeError(f"CSV Line Data Ingestion stream exception hit: {str(e)}")
            
