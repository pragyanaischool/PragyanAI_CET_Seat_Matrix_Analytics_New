import re
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    Advanced Sticky Metadata Look-Ahead State Machine Parser for PragyanAI.
    Locks college profile information in memory and dynamically generates a 
    separate relational record for each individual engineering department extracted.
    """
    def __init__(self):
        # High-capacity model used as a fallback safety layer for highly unstructured files
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        
        self.recovery_template = PromptTemplate.from_template("""
        You are an elite data engineer specializing in structuring educational seat matrices.
        Analyze the following text block from an engineering seat allocation matrix.
        Due to grid constraints, rows or department branches (like Computer Science, Automobile, etc.) 
        might have wrapped incorrectly across multiple lines.
        
        Isolate the active college name, city, district, and address. Then, list ALL individual 
        engineering branches (e.g., Artificial Intelligence, Computer Science, Automobile, etc.) 
        alongside their total intake capacities.
        
        For every single department found, generate a separate object keeping the college metadata identical.
        
        Return the results strictly formatted inside a single valid JSON array of objects:
        [
          {{
            "college_name": "Government Engineering College",
            "city": "Challakere",
            "district": "Chitradurga",
            "address": "BALLARI ROAD CHALLAKERE,CHITRADURGA",
            "dept": "COMPUTER SCIENCE AND ENGINEERING",
            "intake": 60
          }}
        ]
        
        Text Segment to Parse:
        {text_stream}
        """)

    def _parse_via_llm_fallback(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """Heals dense, heavily wrapped multi-line tables using Groq inference frames."""
        try:
            chain = self.recovery_template | self.llm
            response = chain.invoke({"text_stream": raw_text}).content
            
            if "```json" in response:
                json_str = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[-1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            parsed_json = json.loads(json_str)
            df = pd.DataFrame(parsed_json)
            
            if not df.empty:
                df['intake_year'] = int(intake_year)
                df['dept'] = df['dept'].astype(str).str.strip().str.upper()
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Parses plaintext or Markdown streams sequentially. Keeps the parent profile 
        sticky in memory and multiplies rows cleanly depending on the number of departments.
        """
        extracted_records = []
        
        # --- STICKY METADATA STATES ---
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        inside_table = False
        
        # Track last processed department name to catch trailing multi-line wraps
        last_parsed_dept = None
        
        # Clean out double quotes and bold markdown tokens split lines
        lines = [line.strip().replace('"', '').replace('*', '') for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # 1. IDENTIFY COLLEGE METADATA LAYER
            # Matches patterns like: "1 Government Engineering College, Challakere, Chitradurga"
            meta_match = re.match(
                r'^[\s#\|]*(\d+)[\s,\|]+([^,\|]+(?:Registrar|College|Institute|University|VISVESWARIAH|Govt)[^,\|]*)[,\|][\s]*([^,\|]+)[,\|][\s]*([^,\n\|]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No.", "SL.No."]):
                # Lock the parent institutional context keys
                current_college = meta_match.group(2).strip()
                current_city = meta_match.group(3).strip()
                current_district = meta_match.group(4).replace('|', '').strip()
                
                # Reset table window flags and trailing parameters for the new section
                inside_table = False
                last_parsed_dept = None
                continue
                
            # 2. CAPTURE & STICK PHYSICAL ADDRESS
            if "ADDRESS" in line.upper() or "ADDRESS :" in line.upper():
                extracted_addr = line.split(":", 1)[1].replace('|', '').strip() if ":" in line else line.replace("Address", "").strip()
                # Prevent sub-addresses from bleeding or overriding if valid main data is active
                if not current_address or len(extracted_addr) > 5:
                    current_address = extracted_addr
                continue
                
            # 3. SWITCH TABLE SCOPE BOUNDARIES
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No.", "| dept |", "| DEPT"]):
                inside_table = True
                continue
            if any(footer in line for footer in ["Ins Total", "TOTAL:", "InsTotal"]) or line.startswith("---"):
                # Crucial Fix: Do not clear parent metadata variables here; let them stay sticky for the next rows
                inside_table = False
                continue
                
            # 4. ROW MULTIPLICATION LOOP (Generates separate row per department)
            if inside_table or current_college:
                # Match a course line containing an intake capacity (e.g., "1 COMPUTER SCIENCE & ENG 60")
                row_match = re.match(r'^[\s\|]*(\d*)[\s,\|]+([A-Z&\s\(\)\-\/\+\.]+?)[\s,\|]+(\d+)[\s\|]*$', line, re.IGNORECASE)
                
                if row_match:
                    dept_candidate = row_match.group(2).strip().upper()
                    
                    # Prevent headers from accidentally generating fake entries
                    if dept_candidate not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "COURSE", "SL NO", "SL NO.", "TOTAL"]:
                        intake_val = int(row_match.group(3).strip())
                        last_parsed_dept = dept_candidate
                        
                        # Append record, binding the new department to the sticky metadata
                        extracted_records.append({
                            "college_name": current_college,
                            "city": current_city,
                            "district": current_district,
                            "address": current_address,
                            "dept": dept_candidate,
                            "intake": intake_val,
                            "intake_year": int(intake_year)
                        })
                        continue
                
                # Check for implicit/stacked numeric strings following a department section
                num_match = re.findall(r'\b\d+\b', line)
                if num_match and last_parsed_dept and len(line) < 15:
                    intake_val = int(num_match[-1])
                    extracted_records.append({
                        "college_name": current_college,
                        "city": current_city,
                        "district": current_district,
                        "address": current_address,
                        "dept": last_parsed_dept,
                        "intake": intake_val,
                        "intake_year": int(intake_year)
                    })
                    continue
                    
                # Check for text-wrap tails (e.g., "AND ENGINEERING" split onto a new line)
                if re.match(r'^[\s\|]*[A-Z&\s\(\)\-\/\+\.]+[\s\|]*$', line, re.IGNORECASE) and last_parsed_dept:
                    extra_fragment = line.replace('|', '').strip().upper()
                    if extra_fragment not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "TOTAL", "COURSE"]:
                        updated_dept = f"{last_parsed_dept} {extra_fragment}"
                        
                        # Re-bind the fixed name to all records matching the trailing buffer
                        for record in reversed(extracted_records):
                            if record["college_name"] == current_college and record["dept"] == last_parsed_dept:
                                record["dept"] = updated_dept
                        last_parsed_dept = updated_dept

        df_final = pd.DataFrame(extracted_records)
        
        # --- DEEP EXTRACTION CRITICAL OVERRIDE ---
        # If regex missed components or failed to parse, let Llama 3 70B handle structural mapping
        if df_final.empty or df_final['dept'].isna().sum() > (len(df_final) * 0.2):
            df_final = self._parse_via_llm_fallback(raw_text, intake_year)
            
        # --- POST-EXTRACTION DATA MATRICES DEDUPLICATION AND SANITIZATION ---
        if not df_final.empty:
            # 1. Strip rows missing a department entirely
            df_final = df_final.dropna(subset=['dept'])
            df_final = df_final[df_final['dept'].astype(str).str.strip() != '']
            
            # 2. Enforce clean formatting across text lines
            df_final['college_name'] = df_final['college_name'].astype(str).str.strip()
            df_final['city'] = df_final['city'].astype(str).str.strip()
            df_final['district'] = df_final['district'].astype(str).str.strip()
            df_final['dept'] = df_final['dept'].astype(str).str.strip().str.upper()
            
            # 3. Deduplicate rows based on complete primary indicators to avoid branch drop-offs
            df_final = df_final.drop_duplicates(
                subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                keep='first'
            )
            df_final.reset_index(drop=True, inplace=True)
            
        return df_final
