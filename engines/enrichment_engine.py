import re
import json
import concurrent.futures
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

class CollegeEnrichmentEngine:
    """
    Enhanced Fault-Tolerant Federated Knowledge Discovery Engine for PragyanAI.
    Concurrently queries multi-engine web search streams and enforces strict fallback 
    alignment to ensure web connection drops or spelling shifts never break relational joins.
    """
    def __init__(self):
        # Initialize open-web tool wrappers safely inside exception boundaries
        try:
            self.wiki = WikipediaAPIWrapper()
        except Exception:
            self.wiki = None
            
        try:
            self.serp_search = SerpAPIWrapper()
        except Exception:
            self.serp_search = None
            
        try:
            self.ddg_search = DuckDuckGoSearchRun()
        except Exception:
            self.ddg_search = None
            
        # Upgraded to high-capacity low-temperature versatile reasoner model
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    def _query_google_stream(self, query: str) -> str:
        if not self.serp_search:
            return "Google Search Wrapper uninitialized or missing API key tokens."
        try:
            return f"=== GOOGLE SEARCH STREAM RES ===\n{self.serp_search.run(query)}"
        except Exception as e:
            return f"Google Search Stream Bypassed/Failed: {str(e)}"

    def _query_ddg_stream(self, query: str) -> str:
        if not self.ddg_search:
            return "DuckDuckGo Wrapper uninitialized."
        try:
            return f"=== DUCKDUCKGO STREAM RES ===\n{self.ddg_search.run(query)}"
        except Exception as e:
            return f"DuckDuckGo Stream Bypassed/Failed: {str(e)}"

    def _query_wikipedia_stream(self, college_name: str, city: str) -> str:
        if not self.wiki:
            return "Wikipedia Wrapper uninitialized."
        try:
            return f"=== WIKIPEDIA ARTICLE STREAM RES ===\n{self.wiki.run(f'{college_name} {city}')}"
        except Exception as e:
            return f"Wikipedia Stream Bypassed/Failed: {str(e)}"

    def discover_college_details(self, college_name: str, city: str) -> dict:
        """
        Orchestrates parallel multi-threaded federated web searches.
        Guarantees structured relational key returns even during total network failure.
        
        Parameters:
            college_name (str): The parsed baseline institute text key.
            city (str): The isolated local urban node.
            
        Returns:
            dict: Standardized institutional parameters bound exactly to the query keys.
        """
        # Enforce strict text casing boundaries to eliminate key-mismatch risks
        target_college_clean = str(college_name).strip().upper()
        target_city_clean = str(city).strip().upper()

        search_query = f"{target_college_clean} {target_city_clean} official website NAAC Grade NBA Accreditation NIRF Ranking"
        context_accumulator = []

        # --- PHASE 1: ASYNCHRONOUS THREADED BROKERS ---
        # Executes search wrappers across parallel background threads to skip I/O bottlenecks
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_google = executor.submit(self._query_google_stream, search_query)
                future_ddg = executor.submit(self._query_ddg_stream, search_query)
                future_wiki = executor.submit(self._query_wikipedia_stream, target_college_clean, target_city_clean)

                context_accumulator.append(future_google.result())
                context_accumulator.append(future_ddg.result())
                context_accumulator.append(future_wiki.result())
        except Exception as thread_fault:
            context_accumulator.append(f"Threadpool allocation broken: {str(thread_fault)}")

        compiled_search_context = "\n\n".join(context_accumulator)

        # --- PHASE 2: SEMANTIC CONSOLIDATION SYSTEM ---
        template = """
        You are an elite educational intelligence agent parsing web search context logs for administrative records.
        Analyze the provided context streams thoroughly and isolate official verification parameters for: {target_college} located in {target_city}.
        
        Context Input Streams:
        {raw_context}
        
        Extract the following parameters strictly based on facts present inside the text:
        1. website: Provide ONLY the pure string URL path, e.g., "https://www.bmsit.ac.in". Do NOT use markdown brackets or link formatting.
        2. naac_rating: Must be formatted as a standard uppercase grade, e.g., A++, A+, A, B, Not Accredited, or N/A.
        3. nba_accredited: Specify cleanly: Yes, No, or Partially Accredited.
        4. nirf_ranking: Provide the exact current numerical integer placement or ranking band range like "151-200". If not found, output "N/A".
        5. summary: A brief, authoritative 2-sentence structural summary detailing history and focus.
        
        Return the metrics strictly formatted inside a single valid JSON block.
        Do not include any preamble, markdown formatting ticks, or conversational commentary.
        
        Target JSON Format to Output:
        {{
            "website": "https://www.example.edu.in",
            "naac_rating": "A+",
            "nba_accredited": "Partially Accredited",
            "nirf_ranking": "151-200",
            "summary": "Established institution focusing on engineering sciences and technology placement frameworks."
        }}
        """

        try:
            formatted_prompt = template.format(
                target_college=target_college_clean,
                target_city=target_city_clean,
                raw_context=compiled_search_context
            )
            
            llm_output = self.llm.invoke(formatted_prompt).content
            clean_content = llm_output.strip()
            
            # Regular expression boundary extractor isolates the JSON body safely
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                start_idx = clean_content.find("{")
                end_idx = clean_content.rfind("}") + 1
                json_str = clean_content[start_idx:end_idx] if start_idx != -1 else clean_content

            extracted_data = json.loads(json_str)

            # Ensure all schema keys are present and uncorrupted
            for field in ["website", "naac_rating", "nba_accredited", "nirf_ranking", "summary"]:
                if field not in extracted_data or not str(extracted_data[field]).strip():
                    extracted_data[field] = "N/A"

        except Exception as e:
            # Secure default fallback instantiation if formatting boundaries or APIs drop
            print(f"[Enrichment Framework Recovered] Network/Format drop handled: {str(e)}")
            extracted_data = {
                "website": "N/A",
                "naac_rating": "N/A",
                "nba_accredited": "N/A",
                "nirf_ranking": "N/A",
                "summary": "Institutional data recorded cleanly. Web infrastructure lookup skipped."
            }

        # 🎯 THE ANTI-COLLAPSE LOCK:
        # Re-bind the exact clean input keys to guarantee a perfect relational pd.merge join match
        extracted_data["college_name"] = target_college_clean
        extracted_data["city"] = target_city_clean
        
        return extracted_data
