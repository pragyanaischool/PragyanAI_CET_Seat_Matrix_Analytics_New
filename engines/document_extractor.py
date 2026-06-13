import io
import os
import pandas as pd
from docx import Document

# --- AI Extraction Engines Import Guard Boundaries ---
try:
    from docling.document_converter import DocumentConverter as IBMDoclingConverter
except ImportError:
    IBMDoclingConverter = None

try:
    from marker.convert import convert_single_pdf as MarkerPDFConverter
    from marker.models import load_all_models as load_marker_models
except ImportError:
    MarkerPDFConverter = None

try:
    from unstructured.partition.pdf import partition_pdf as UnstructuredPDFPartitioner
except ImportError:
    UnstructuredPDFPartitioner = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


class PragyanDocumentExtractor:
    """
    Enterprise ensemble data extraction and normalization engine for PragyanAI.
    Concurrently leverages IBM's Docling, Marker, and Unstructured to transform
    highly unpredictable administrative PDFs into structured, layout-healed markdown text streams.
    """
    
    @staticmethod
    def extract_to_text_stream(file_bytes: bytes, file_name: str) -> str:
        """
        Routes incoming binaries to specific data parsing handlers based on file type.
        
        Parameters:
            file_bytes (bytes): Raw binary payload of the uploaded document file.
            file_name (str): Original string file name containing the explicit extension suffix.
            
        Returns:
            str: Normalized plain-text/markdown string data stream.
        """
        if not file_name or "." not in file_name:
            raise ValueError("Invalid document reference: Missing explicit format extension suffix.")
            
        ext = file_name.split(".")[-1].lower()
        
        if ext == "pdf":
            return PragyanDocumentExtractor._extract_pdf_ensemble(file_bytes)
        elif ext in ["xlsx", "xls"]:
            return PragyanDocumentExtractor._extract_spreadsheet(file_bytes)
        elif ext == "csv":
            return PragyanDocumentExtractor._extract_csv(file_bytes)
        elif ext in ["docx", "doc"]:
            return PragyanDocumentExtractor._extract_docx(file_bytes)
        else:
            raise ValueError(f"Ingestion Core Aborted: Unsupported file format specification: .{ext}")

    @staticmethod
    def _extract_pdf_ensemble(file_bytes: bytes) -> str:
        """
        Executes a prioritized fallback parsing pipeline to reconstruct text alignments
        without dropping columns, text lines, or clipping table dimensions.
        """
        extracted_results = {}
        temp_pdf_name = "temp_operational_ingestion.pdf"
        
        # Write bytes cleanly to disk temporarily to accommodate file-path-bound ML libraries
        with open(temp_pdf_name, "wb") as f:
            f.write(file_bytes)
            
        try:
            # --- STRATEGY 1: IBM DOCLING (Enterprise layout object-detection) ---
            if IBMDoclingConverter is not None:
                try:
                    converter = IBMDoclingConverter()
                    docling_doc = converter.convert_to_markdown(temp_pdf_name)
                    extracted_results["docling"] = docling_doc.markdown
                except Exception as e:
                    print(f"[Ensemble Alert] IBM Docling parsing phase bypassed: {str(e)}")

            # --- STRATEGY 2: MARKER PDF (Optimized for text, grids, and formulas) ---
            if MarkerPDFConverter is not None:
                try:
                    model_lst, metadata_ctx = load_marker_models()
                    marker_text, _, _ = MarkerPDFConverter(temp_pdf_name, model_lst, metadata_ctx)
                    extracted_results["marker"] = marker_text
                except Exception as e:
                    print(f"[Ensemble Alert] Marker layout transformation phase bypassed: {str(e)}")

            # --- STRATEGY 3: UNSTRUCTURED (Vision-augmented chunking engine) ---
            if UnstructuredPDFPartitioner is not None:
                try:
                    elements = UnstructuredPDFPartitioner(
                        filename=temp_pdf_name,
                        strategy="hi_res",  # Triggers deeper vision model inference
                        infer_table_structure=True
                    )
                    unstructured_text = "\n".join([str(el) for el in elements])
                    extracted_results["unstructured"] = unstructured_text
                except Exception as e:
                    print(f"[Ensemble Alert] Unstructured pipeline chunking phase bypassed: {str(e)}")

            # --- STRATEGY 4: SPATIAL LOCAL ENFORCER (pdfplumber backup) ---
            # Activates if all high-level ML dependencies fail to initiate or process
            if not extracted_results:
                if pdfplumber is not None:
                    print("[Ensemble Warning] Deep learning architectures bypassed. Running pdfplumber layout fallback...")
                    text_accum = []
                    with pdfplumber.open(temp_pdf_name) as pdf:
                        for idx, page in enumerate(pdf.pages):
                            page_text = page.extract_text(layout=True)
                            if page_text:
                                text_accum.append(f"\n--- PAGE {idx + 1} ---\n{page_text}")
                    return "\n".join(text_accum)
                else:
                    raise RuntimeError("Critical System Fault: No active document parser extensions found inside local runtime environment.")

            # --- PIPELINE CONSOLIDATION & RECONCILIATION SELECTION ---
            # Prioritize clean Markdown frameworks to ensure the Downstream state machine is reliable
            if "docling" in extracted_results and extracted_results["docling"].strip():
                return extracted_results["docling"]
            elif "marker" in extracted_results and extracted_results["marker"].strip():
                return extracted_results["marker"]
            else:
                # Fall back directly to the oldest element inside the captured dictionary logs
                return list(extracted_results.values())[0]
                
        finally:
            # Housekeeping thread cleanup loop to ensure local disk storage remains pristine
            if os.path.exists(temp_pdf_name):
                try:
                    os.remove(temp_pdf_name)
                except Exception:
                    pass

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        """Flattens paragraphs and multi-column tables from Word documents."""
        try:
            doc = Document(io.BytesIO(file_bytes))
            text_accum = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_accum.append(paragraph.text.strip())
                    
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
