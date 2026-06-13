import re
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    Upgraded Sticky Hybrid Parsing State Machine for PragyanAI.
    Detects spatial signature violations (like lumped department strings or pincode intakes)
    and routes them to llama-3.3-70b-versatile to split data into clean individual rows.
    """
    def __init__(self):
        # High-capacity versatile model to handle complex layout parsing and formatting recovery
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        
        # Comprehensive Zero-Shot recovery matrix layout healing framework template
        self.recovery_template = PromptTemplate.from_template("""
        You are an elite data engineering extraction agent specialized in structural matrix normalization.
        Analyze the following text block from an engineering seat allocation document. The input may contain 
        complex layouts, wrapped text rows, or raw Markdown grid tables with vertical boundary pipes (|).
        
        Isolate the active college name, city, district, and physical address. Then, extract ALL individual 
        engineering branches/departments alongside their total intake capacity numbers.
        
        For every single department found, generate a separate object keeping the college metadata completely identical.
        
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
          }}
        ]
        
        Text Segment to Parse:
        {text_stream}
        """)

    def _parse_via_llm_fallback(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """Heals dense, heavily wrapped markdown tables using specialized string isolation."""
        try:
            chain = self.recovery_template | self.llm
            response = chain.invoke({"text_stream": raw_text}).content
            
            clean_content = response.strip()
            
            # Robust boundary handler sweeps and clips out markdown tags to isolate the raw JSON array bounds
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
                df['intake_year'] = int(intake_year)
                if 'dept' in df.columns:
                    df['dept'] = df['dept'].astype(str).str.strip().str.upper()
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Parses plaintext or Markdown streams sequentially using a look-ahead state machine.
        Keeps parent institution variables locked in memory and generates a distinct row 
        for each separate academic track discovered.
        """
        extracted_records = []
        
        # --- STICKY METADATA STATE MEMORY CAPSULES ---
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        inside_table = False
        
        # Track the last successfully parsed track to capture split-line trailing blocks
        last_parsed_dept = None
        
        # Clean out quotes and markdown bold markers to prevent formatting variants
        lines = [line.strip().replace('"', '').replace('*', '') for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # Skip pure markdown grid spacer lines (e.g., |---|---|---|)
            if re.match(r'^[\s\|:\-]+$', line):
                continue

            # 1. METADATA LAYER: Detect campus profile indices and geographic boundaries
            meta_match = re.search(
                r'(?:^|\|)[\s#]*(\d+)[\s,\|]+([^,\|]+(?:Registrar|College|Institute|University|VISVESWARIAH|Govt)[^,\|]*)[,\|][\s]*([^,\|]+)[,\|][\s]*([^,\n\|]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No.", "SL.No."]):
                # Lock target variables into sticky context states
                current_college = meta_match.group(2).strip()
                current_city = meta_match.group(3).strip()
                current_district = meta_match.group(4).replace('|', '').strip()
                
                # Reset table boundary scopes and look-ahead tracks for the new section
                inside_table = False
                last_parsed_dept = None
                continue
                
            # 2. ADDRESS SEPARATION: Extracts physical coordinates
            if "ADDRESS" in line.upper() or "ADDRESS :" in line.upper():
                extracted_addr = line.split(":", 1)[1].replace('|', '').strip() if ":" in line else line.replace("Address", "").strip()
                if not current_address or len(extracted_addr) > 5:
                    current_address = extracted_addr
                continue
                
            # 3. SWITCH TABLE SCOPE REGIONS
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No.", "dept", "DEPT", "Intake", "| dept |"]):
                inside_table = True
                continue
            if any(footer in line for footer in ["Ins Total", "TOTAL:", "InsTotal"]) or line.startswith("---"):
                inside_table = False
                continue
                
            # 4. DETERMINISTIC DYNAMIC ROW MULTIPLICATION LOOP
            if inside_table or current_college:
                row_match = re.match(r'^[\s\|]*(\d*)[\s,\|]*([A-Z&\s\(\)\-\/\+\.\b]+?)[\s,\|]+(\d+)[\s\|]*$', line, re.IGNORECASE)
                
                if row_match:
                    dept_candidate = row_match.group(2).strip().upper()
                    
                    if dept_candidate not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "COURSE", "SL NO", "SL NO.", "TOTAL", "INTAKE"]:
                        intake_val = int(row_match.group(3).strip())
                        last_parsed_dept = dept_candidate
                        
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
                
                # Look-Ahead: Check for numeric capacity items trailing directly after active sections
                num_match = re.findall(r'\b\d+\b', line)
                if num_match and last_parsed_dept and len(line.replace('|', '').strip()) < 15:
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
                    
                # Look-Ahead: Check for text-wrap tails (e.g., "AND ENGINEERING" split onto a new line)
                if re.match(r'^[\s\|]*[A-Z&\s\(\)\-\/\+\.]+[\s\|]*$', line, re.IGNORECASE) and last_parsed_dept:
                    extra_fragment = line.replace('|', '').strip().upper()
                    if extra_fragment not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "TOTAL", "COURSE", "INTAKE"]:
                        updated_dept = f"{last_parsed_dept} {extra_fragment}"
                        
                        for record in reversed(extracted_records):
                            if record["college_name"] == current_college and record["dept"] == last_parsed_dept:
                                record["dept"] = updated_dept
                        last_parsed_dept = updated_dept

        df_final = pd.DataFrame(extracted_records)
        
        # --- CRITICAL SPATIAL EXTRACTION SIGNATURE VIOLATION INTERCEPTOR ---
        # Scans the dataframe to see if multiple separate records were lumped together into single cells
        is_signature_violated = False
        if not df_final.empty:
            for dept_str in df_final['dept'].astype(str):
                # If a department string contains multiple independent fields/disciplines or keywords
                if len(dept_str) > 75 or dept_str.count("ENGINEERING") > 1 or "SEATS" in dept_str:
                    is_signature_violated = True
                    break
            for intake_val in df_final['intake']:
                # If intake numbers look like an architectural pincode or zip layout field
                if intake_val > 1000:
                    is_signature_violated = True
                    break

        # If regular expressions returned an empty frame or failed to slice clean rows,
        # route execution to the upgraded versatile fallback model to extract separate records cleanly.
        if df_final.empty or is_signature_violated or df_final['dept'].isna().sum() > (len(df_final) * 0.11):
            df_final = self._parse_via_llm_fallback(raw_text, intake_year)
            
        # --- POST-EXTRACTION DATA MATRICES SANITIZATION AND DEDUPLICATION ---
        if not df_final.empty:
            df_final = df_final.dropna(subset=['dept'])
            df_final = df_final[df_final['dept'].astype(str).str.strip() != '']
            
            df_final['college_name'] = df_final['college_name'].astype(str).str.strip()
            df_final['city'] = df_final['city'].astype(str).str.strip()
            df_final['district'] = df_final['district'].astype(str).str.strip()
            if 'dept' in df_final.columns:
                df_final['dept'] = df_final['dept'].astype(str).str.strip().str.upper()
            
            df_final = df_final.drop_duplicates(
                subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                keep='first'
            )
            df_final.reset_index(drop=True, inplace=True)
            
        return df_final
