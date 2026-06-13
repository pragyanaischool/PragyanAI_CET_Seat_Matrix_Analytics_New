import re
import json
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

class CETSeatMatrixParser:
    """
    Upgraded fault-tolerant hybrid parsing state machine for PragyanAI.
    Adapts dynamically to structured Markdown layouts produced by ensemble libraries,
    incorporating a zero-shot LLM recovery loop alongside automated data sanitization
    and structural deduplication routines to ensure data lake accuracy.
    """
    def __init__(self):
        # High-capacity Llama 3 instance to resolve layout distortions
        # Temperature is set to 0.0 to prevent hallucinations and enforce strict accuracy
        self.llm = ChatGroq(model_name="llama3-70b-8192", temperature=0.0)
        
        # Explicit Prompt Template to extract data structures from complex sections
        self.recovery_template = PromptTemplate.from_template("""
        You are an elite data engineering extraction agent parsing structured educational data layouts.
        Analyze the following text chunk extracted from an engineering seat matrix file page. 
        The layout parsing may be structurally complex, unformatted, or misaligned due to visual row wrapping.
        
        Isolate the college name, municipal city, geographic district, full physical address, 
        and list all engineering branches/departments alongside their corresponding total intake capacities.
        Ignore administrative quotas, percentage distributions, and summary row calculations.
        
        Return the results strictly formatted inside a single valid JSON array of objects. 
        Do not add any preamble, conversational commentary, or trailing notes. 
        Each object in the array must follow this schema format exactly:
        [
          {{
            "college_name": "Full official name of the college",
            "city": "City",
            "district": "District",
            "address": "Full physical verification address",
            "dept": "CLEAN UPPERCASE DEPARTMENT NAME",
            "intake": 60
          }}
        ]
        
        Unstructured Text Stream Chunk to Heal:
        {text_stream}
        """)

    def _parse_via_llm_fallback(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Activates as a fallback engine to restructure complex text segments 
        using zero-shot structural extraction loops on Groq frames.
        """
        try:
            chain = self.recovery_template | self.llm
            response = chain.invoke({"text_stream": raw_text}).content
            
            # Extract JSON payload cleanly if wrapped in markdown formatting blocks
            if "```json" in response:
                json_str = response.split("```json")[-1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[-1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            parsed_json = json.loads(json_str)
            df = pd.DataFrame(parsed_json)
            
            if not df.empty:
                # Stamp the correct target financial year onto every recovered record row
                df['intake_year'] = int(intake_year)
                # Enforce clean uppercase nomenclature for the database layer
                df['dept'] = df['dept'].astype(str).str.strip().str.upper()
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def parse_text_stream(self, raw_text: str, intake_year: int) -> pd.DataFrame:
        """
        Parses multi-page text or Markdown streams sequentially using a state machine layout,
        automatically performing semantic cleaning and duplicate dropping.
        
        Parameters:
            raw_text (str): Flattened plaintext or Markdown input stream from document extractors.
            intake_year (int): Academic logging time horizon variable (e.g., 2024, 2026).
            
        Returns:
            pd.DataFrame: Sanitized, completely unique relational rows matching the SQLite schema.
        """
        extracted_records = []
        
        # State tracking identifiers
        current_college = None
        current_city = None
        current_district = None
        current_address = None
        inside_table = False
        
        # Split stream into distinct lines, stripping quotes and bold markdown artifacts (**)
        lines = [line.strip().replace('"', '').replace('*', '') for line in raw_text.split('\n') if line.strip()]
        
        for line in lines:
            # 1. METADATA MATCHING: Detects core campus names and locations
            meta_match = re.match(
                r'^[\s#\|]*(\d+)[\s,\|]+([^,\|]+(?:Registrar|College|Institute|University|VISVESWARIAH|Govt)[^,\|]*)[,\|][\s]*([^,\|]+)[,\|][\s]*([^,\n\|]+)', 
                line, 
                re.IGNORECASE
            )
            
            if meta_match and not any(k in line for k in ["Address", "Intake", "Total", "Sl.No.", "SL.No."]):
                current_college = meta_match.group(2).strip()
                current_city = meta_match.group(3).strip()
                current_district = meta_match.group(4).replace('|', '').strip()  # Clean trailing markdown boundaries
                inside_table = False
                continue
                
            # 2. ADDRESS SEPARATION: Extracts physical postal paths
            if "ADDRESS" in line.upper() or "ADDRESS :" in line.upper():
                current_address = line.split(":", 1)[1].replace('|', '').strip() if ":" in line else line.replace("Address", "").strip()
                continue
                
            # 3. BOUNDARY CONDITION SWITCHES: Tracks entry and exit for active course tables
            if any(header in line for header in ["Course Name", "Total Intake", "Sl.No.", "SL.No.", "| dept |", "| DEPT"]):
                inside_table = True
                continue
            if any(footer in line for footer in ["Ins Total", "TOTAL:", "InsTotal"]) or line.startswith("---"):
                inside_table = False
                continue
                
            # 4. DETERMINISTIC ROW PARSING: Processes engineering tracks and seat counts within active windows
            if inside_table:
                # Regular expression strips enclosing pipeline separators (|) generated by Markdown tables
                row_match = re.match(r'^[\s\|]*(\d+)[\s,\|]+([A-Z&\s\(\)\-\/]+)[\s,\|]+(\d+)', line, re.IGNORECASE)
                
                if row_match:
                    dept_candidate = row_match.group(2).strip().upper()
                    # Filter out table header text variations caught in active windows
                    if dept_candidate not in ["COURSE NAME", "TOTAL INTAKE", "DEPT", "COURSE", "SL NO", "SL NO."]:
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

        df_res = pd.DataFrame(extracted_records)
        
        # --- HYBRID RECOVERY INTERCEPTOR TRIGGER ---
        # If the input text uses a complex format that breaks standard regex loops, 
        # the system routes the data to the Llama 3 70B recovery layout healer automatically.
        if df_res.empty:
            df_res = self._parse_via_llm_fallback(raw_text, intake_year)
            
        # --- COMPREHENSIVE POST-EXTRACTION CLEANING & DEDUPLICATION PIPELINE ---
        if not df_res.empty:
            # Step A: Drop rows missing a department/course branch name entirely (removes visual noise and misalignments)
            df_res = df_res.dropna(subset=['dept'])
            df_res = df_res[df_res['dept'].astype(str).str.strip() != '']
            
            # Step B: Normalize string contents to prevent formatting variances from causing false duplicates
            df_res['college_name'] = df_res['college_name'].astype(str).str.strip()
            df_res['city'] = df_res['city'].astype(str).str.strip()
            df_res['district'] = df_res['district'].astype(str).str.strip()
            df_res['dept'] = df_res['dept'].astype(str).str.strip().str.upper()
            
            # Step C: Drop rows where all primary schema indicators match perfectly
            df_res = df_res.drop_duplicates(
                subset=['college_name', 'city', 'district', 'dept', 'intake', 'intake_year'],
                keep='first'
            )
            
            # Reset indices for a clean downstream presentation layer
            df_res.reset_index(drop=True, inplace=True)
            
        return df_res
