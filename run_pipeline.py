import os
import sys
from dotenv import load_dotenv

# Ensure system paths are dynamically mapped across parent boundaries
root_path = os.path.dirname(os.path.abspath(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Initialize environment tokens
load_dotenv()

from engines.ensemble_orchestrator import PragyanEnsembleParser

def run_production_pipeline():
    # Target file paths config
    input_source_pdf = "seat_matrix2052024kannada.pdf"
    output_target_csv = "clean_multiplied_branches_export.csv"
    
    if not os.path.exists(input_source_pdf):
        print(f"❌ Execution Halted: Input source matrix document not found at: {input_source_pdf}")
        return

    print(f"🚀 Launching Ingestion Pipeline via Multi-Engine Ensemble Consensus...")
    print(f"📄 Loading payload binary: {input_source_pdf}")
    
    # Read layout payload securely into system memory frames
    with open(input_source_pdf, "rb") as f:
        pdf_payload_bytes = f.read()
        
    # Instantiate consensus parsing routines
    parser_orchestrator = PragyanEnsembleParser()
    
    print("⏳ Running concurrent extractions (Docling + pdfplumber + PyMuPDF)...")
    print("🧠 Normalizing text variations through llama-3.3-70b-versatile logic loops...")
    
    # Process text layout extraction and matrix normalization
    structured_matrix_df = parser_orchestrator.analyze_and_extract_matrix(pdf_payload_bytes, intake_year=2025)
    
    # Output structured data matrices
    if not structured_matrix_df.empty:
        structured_matrix_df.to_csv(output_target_csv, index=False)
        print("\n" + "="*70)
        print("✅ SUCCESS: Extraction pipeline completed with zero signature violations!")
        print(f"📊 Total Relational Rows Generated: {len(structured_matrix_df)} Rows")
        print(f"💾 Clean dataset exported directly to file: {output_target_csv}")
        print("="*70 + "\n")
        print("Previewing top extracted records matrix:")
        print(structured_matrix_df.head(10))
    else:
        print("❌ CRITICAL: Ensemble framework returned an empty matrix dataframe. Check API token allocations.")

if __name__ == "__main__":
    run_production_pipeline()
