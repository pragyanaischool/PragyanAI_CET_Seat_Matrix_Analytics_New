import re
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    An enterprise-grade hybrid state-machine parser for the Karnataka CET Seat Matrix.
    Combines flexible, fault-tolerant Regular Expressions with an LLM fallback loop
    via Groq to guarantee clean relational parsing across shifting document layouts.
    """
    def __init__(self):
        # High-capacity Llama 3 model for healing complex, multi-line broken blocks
        # Temperature is pinned to 0.0 to guarantee factual formatting precision
        self.llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.0)
        
        self.fallback_template = PromptTemplate.from_template("""
        You are an expert data sanitization assistant engineering educational data lake structures.
        Analyze the following broken text chunk from a seat matrix document page layout. 
        Rows might have wrapped incorrectly across lines due to grid alignment shifts.
        
        Extract the engineering departments/branches and their corresponding total intake capacities. 
        Ignore percentage allotments, administrative quotas, and total calculation summaries.
        
        Return the structured rows strictly inside a single valid JSON array of objects. 
        Do not add any conversational text, explanation, or markdown codeblocks. 
        Each object must follow this format:
        [
            {{"dept": "CLEAN UPPERCASE DEPARTMENT NAME", "intake": 60}},
            {{"dept": "ANOTHER COURSE SPECIALTY", "intake": 120}}
        ]
        
        Text Segment:
        {text_chunk}
        """)
        
    def _clean_chunk_with_llm(self, broken_text_chunk: str) -> list:
        try:
            chain = self.fallback_template | self.llm
            response = chain.invoke({"text_chunk": broken_text_chunk}).content
            
            # Extract JSON string safely if wrapped in markdown formatting tags
            if "```json" in response:
                json_str = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[-1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            import json
            parsed_json = json.loads(json_str)
            if isinstance(parsed_json, list):
                return parsed_json
        except Exception:
            pass
        return []

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        extracted_records = []
        
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        
        inside_table = False
        unresolved_buffer = []  # Accumulates un-parsed multi-line fragments
        
        # Split stream into clear lines and sanitize surrounding quote artifacts
        lines = [line.strip().replace('"', '') for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # 1. METADATA MATCHING: Captures patterns like "1 Government Engineering College, Challakere, Chitradurga"
            # Upgraded pattern uses non-greedy matches and supports pipe (|) or comma dividers seamlessly
            meta_match = re.match(
                r'^[\s]*(\d+)[\s,\|]+([^,\|]+(?:Registrar|College|Institute|University|VISVESWARIAH|Govt)[^,\|]*)[,\|][\s]*([^,\|]+)[,\|][\s]*([^,\n\|]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No.", "SL.No."]):
                # Flush the unresolved data buffer before switching state context to a new college block
                if unresolved_buffer:
                    extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))
                    unresolved_buffer = []
                
                current_college = meta_match.group(2).strip()
                current_city = meta_match.group(3).strip()
                current_district = meta_match.group(4).strip()
                inside_table = False
                continue
                
            # 2. ADDRESS MATCHING: Retains locations across data scopes
            if "ADDRESS" in line.upper():
                current_address = line.split(":", 1)[1].strip() if ":" in line else line.replace("Address", "").strip()
                continue
                
            # 3. BOUNDARY MATRIX LOCKS: Sets active evaluation window coordinates
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No.", "COURSE NAME"]):
                inside_table = True
                continue
            if any(footer in line for footer in ["Ins Total", "TOTAL:", "InsTotal"]):
                if unresolved_buffer:
                    extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))
                    unresolved_buffer = []
                inside_table = False
                continue
                
            # 4. DATA ROW EXTRACTION: Isolates academic disciplines inside active boundaries
            if inside_table:
                # Upgraded Regex: Handles space delimiters, csv commas, and text pipeline bounds (|) smoothly
                row_match = re.match(r'^[\s]*(\d+)[\s,\|]+([A-Z&\s\(\)\-\/]+)[\s,\|]+(\d+)', line, re.IGNORECASE)
                
                if row_match:
                    dept_candidate = row_match.group(2).strip().upper()
                    # Prevent headers or labels from being processed as data fields
                    if dept_candidate not in ["COURSE NAME", "TOTAL INTAKE", "SL NO", "SL NO."]:
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
                    # Line layout is complex or multi-line wrapped; cache in structural buffer for LLM processing
                    unresolved_buffer.append(line)
                    
        # Final cleanup sweep to clear remaining data blocks from the buffer
        if unresolved_buffer:
            extracted_records.extend(self._process_buffer(unresolved_buffer, current_college, current_city, current_district, current_address, intake_year))

        return pd.DataFrame(extracted_records)

    def _process_buffer(self, buffer_lines: list, college: str, city: str, district: str, address: str, year: int) -> list:
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
