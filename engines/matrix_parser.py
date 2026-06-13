import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    A robust hybrid state-machine parser for the Karnataka CET Seat Matrix text dumps.
    Combines efficient, deterministic Regular Expressions with a Large Language Model
    (LLM) fallback via Groq to cleanly extract and structure data from messy tables.
    """
    def __init__(self):
        # Initializing the high-capacity Llama 3 model for structural analysis
        # Temperature is set to 0.0 to prevent hallucinations and enforce strict accuracy
        self.llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.0)
        
        # LLM prompt template to heal and condense complex, multi-line broken blocks
        self.fallback_template = PromptTemplate.from_template("""
        You are an expert data cleaning assistant specializing in educational admissions documents.
        Analyze the following broken text chunk extracted from a seat matrix table where row data might have wrapped incorrectly across lines.
        
        Extract the academic departments and their corresponding total intake numbers. 
        Ignore administrative quotas, percentage counts, and summary statistics.
        
        Return the structured rows strictly inside a JSON array of objects. Do not add any conversational text or explanation. 
        Each object must strictly match this format:
        [
            {{"dept": "CLEAN UPPERCASE DEPARTMENT NAME", "intake": 60}},
            {{"dept": "ANOTHER DEPARTMENT NAME", "intake": 120}}
        ]
        
        Broken Text Segment:
        {text_chunk}
        """)
        
    def _clean_chunk_with_llm(self, broken_text_chunk: str) -> list:
        """
        Fallback routine: Uses an LLM to resolve text wraps (e.g., 'ARTIFICIAL INTELLIGENCE AND MACHINE' 
        on one line and 'LEARNING' + metrics on subsequent lines).
        """
        try:
            chain = self.fallback_template | self.llm
            response = chain.invoke({"text_chunk": broken_text_chunk}).content
            
            # Clean possible markdown wrap formatting elements from the LLM's JSON response
            json_str = response.split("```json")[-1].split("```")[0].strip()
            
            import json
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
        except Exception as e:
            # Silently fall back to an empty array if an inference exception hits
            pass
        return []

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Parses an entire multi-page text stream sequentially using a state machine.
        
        Parameters:
            raw_text (str): Raw multi-line text dump extracted from the ingestion engine.
            intake_year (int): Academic timeline tracking variable (e.g., 2024, 2025).
            
        Returns:
            pd.DataFrame: Relational rows mapped directly to the master SQLite schema.
        """
        extracted_records = []
        
        # State tracking variables
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        
        inside_table = False
        unresolved_buffer = []  # Accumulates problematic table chunks for LLM evaluation
        
        # Splitting input stream into clean line fragments
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # 1. METADATA MATCHING: Captures patterns like "1 Government Engineering College, Challakere, Chitradurga"
            meta_match = re.match(
                r'^["\s]*(\d+)\s+([^,]+ Registrar|[^,]+ College|[^,]+ Institute[^,]*|VISVESWARIAH[^,]*|Govt[^,]*|University[^,]*),\s*([^,]+),\s*([^,\n"]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No."]):
                # Flush any pending trailing table buffer elements before shifting context to a new college
                if unresolved_buffer:
                    extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))
                    unresolved_buffer = []
                
                current_college = meta_match.group(2).replace('"', '').strip()
                current_city = meta_match.group(3).replace('"', '').strip()
                current_district = meta_match.group(4).replace('"', '').strip()
                inside_table = False
                continue
                
            # 2. ADDRESS MATCHING: Captures physical location descriptors
            if "ADDRESS" in line.upper() or "ADDRESS :" in line.upper():
                current_address = line.split(":", 1)[1].replace('"', '').strip() if ":" in line else line.replace("Address", "").strip()
                continue
                
            # 3. BOUNDARY LOCKS: Identifies the start and end of a course table area
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No."]):
                inside_table = True
                continue
            if "Ins Total" in line or "TOTAL:" in line:
                if unresolved_buffer:
                    extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))
                    unresolved_buffer = []
                inside_table = False
                continue
                
            # 4. ROW EXTRACTION: Processes rows within active table blocks
            if inside_table:
                # Regular regex try: Expects clean, unbroken formatting: [Index] + [Department Text] + [Numbers]
                row_match = re.match(r'^["\s]*(\d+)["\s]*,?["\s]*([A-Z&\s\(\)\-\/]+)\s*["\s,\|]+(\d+)', line, re.IGNORECASE)
                
                if row_match:
                    dept_candidate = row_match.group(2).strip().upper()
                    # Filter out header variations or misclassifications
                    if dept_candidate not in ["COURSE NAME", "TOTAL INTAKE"]:
                        intake_val = int(row_match.group(3).strip())
                        extracted_records.append({
                            "college_name": current_college,
                            "city": current_city,
                            "district": current_district,
                            "address": current_address,
                            "dept": dept_candidate,
                            "intake": intake_val,
                            "intake_year": int(intake_year)
                        })
                else:
                    # If regex fails to unpack the line (e.g., text wraps over multiple lines), 
                    # send it to the buffer for consolidated LLM parsing
                    unresolved_buffer.append(line)
                    
        # Final catch-all sweep to clear out remaining items from the buffer
        if unresolved_buffer:
            extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))

        return pd.DataFrame(extracted_records)

    def _process_buffer(self, buffer_lines: list, college: str, city: str, district: str, address: str, year: int) -> list:
        """
        Helper method that aggregates broken text lines, passes them to the LLM, 
        and maps the response objects back into unified database records.
        """
        records = []
        if not college or not buffer_lines:
            return records
            
        compiled_block = "\n".join(buffer_lines)
        llm_results = self._clean_chunk_with_llm(compiled_block)
        
        for item in llm_results:
            if "dept" in item and "intake" in item:
                records.append({
                    "college_name": college,
                    "city": city,
                    "district": district,
                    "address": address,
                    "dept": str(item["dept"]).strip().upper(),
                    "intake": int(item["intake"]),
                    "intake_year": int(year)
                })
        return records
