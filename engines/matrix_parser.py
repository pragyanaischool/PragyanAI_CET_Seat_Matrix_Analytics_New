import re
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    Advanced Sticky Metadata Look-Ahead State Machine Parser for PragyanAI.
    Natively parses markdown tables and structural text streams while keeping institutional
    metadata sticky in memory to multiply department rows cleanly.
    """
    def __init__(self):
        # High-capacity upgraded versatile model to heal highly fragmented or markdown-heavy matrix structures
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)
        
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
            
            # Extract pure JSON arrays if the model returned markdown codeblocks
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
        except Exception as e:
            print(f"[Fallback Engine Info] Alternate parsing window skipped: {str(e)}")
        return pd.DataFrame()

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Parses plain text streams and Markdown grid rows sequentially. Keeps institutional
        metadata sticky in memory and multiplies rows per department discipline.
        """
        extracted_records = []
        
        # --- STICKY METADATA STATES ---
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        inside_table = False
        
        last_parsed_dept = None
        
        # Split stream into distinct lines, stripping double quotes and bold markdown tokens (**)
        lines = [line.strip().replace('"', '').replace('*', '') for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # Skip pure Markdown structural separators (e.g., |---|---|---|)
            if re.match(r'^[\s\|:\-]+$', line):
                continue

            # 1. IDENTIFY COLLEGE METADATA LAYER
            # Enhanced regex to capture college rows even when wrapped inside markdown table cells or vertical pipes
            meta_match = re.search(
                r'(?:^|\|)[\s#]*(\d+)[\s,\|]+([^,\|]+(?:Registrar|College|Institute|University|VISVESWARIAH|Govt)[^,\|]*)[,\|][\s]*([^,\|]+)[,\|][\s]*([^,\n\|]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No.", "SL.No."]):
                current_college = meta_match.group(2).strip()
                current_city = meta_match.group(3).strip()
                current_district = meta_match.group(4).replace('|', '').strip()
                
                inside_table = False
                last_parsed_dept = None
                continue
                
            # 2. CAPTURE & STICK PHYSICAL ADDRESS
            if "ADDRESS" in line.upper():
                extracted_addr = line.split(":", 1)[1].replace('|', '').strip() if ":" in line else line.replace("Address", "").strip()
                if not current_address or len(extracted_addr) > 5:
                    current_address = extracted_addr
                continue
                
            # 3. SWITCH TABLE SCOPE BOUNDARIES
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No.", "dept", "DEPT", "Intake"]):
                inside_table = True
                continue
            if any(footer in line for footer in ["Ins Total", "TOTAL:", "InsTotal"]) or line.startswith("---"):
                inside_table = False
                continue
                
            # 4. ROW MULTIPLICATION LOOP
            if inside_table or current_college:
                # Extract values out of markdown vertical pipe grids (|)
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
                
                # Check for implicit/stacked numeric fields following an active department block
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
                    
                # Check for text-wrap tails (e.g., handles wrapped blocks smoothly)
                if re.match(r'^[\s\|]*[A-Z&\s\(\)\-\/\+\.]+[\s\|]*$', line, re.IGNORECASE) and last_parsed_dept:
                    extra_fragment = line.replace('|', '').strip().upper()
                    if extra_fragment not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "TOTAL", "COURSE", "INTAKE"]:
                        updated_dept = f"{last_parsed_dept} {extra_fragment}"
                        
                        for record in reversed(extracted_records):
                            if record["college_name"] == current_college and record["dept"] == last_parsed_dept:
                                record["dept"] = updated_dept
                        last_parsed_dept = updated_dept

        df_final = pd.DataFrame(extracted_records)
        
        # --- COMPREHENSIVE OVERRIDE INTERCEPTOR ---
        # If regular expressions returned an empty frame, process via upgraded versatile fallback model
        if df_final.empty or df_final['dept'].isna().sum() > (len(df_final) * 0.1):
            df_final = self._parse_via_llm_fallback(raw_text, intake_year)
            
        # --- POST-EXTRACTION DATA MATRICES DEDUPLICATION ---
        if not df_final.empty:
            df_final = df_final.dropna(subset=['dept'])
            df_final = df_final[df_final['dept'].astype(str).str.strip() != '']
            
            df_final['college_name'] = df_final['college_name'].astype(str).str.strip()
            df_final['city'] = df_final['city'].astype(str).str.strip()
            df_final['district'] = df_final['district'].astype(str).str.strip()
            df_final['dept'] = df_final['dept'].astype(str).str.strip().str.upper()
            
            df_final = df_final.drop_duplicates(
                subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                keep='first'
            )
            df_final.reset_index(drop=True, inplace=True)
            
        return df_final
