import re
import json
import concurrent.futures
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

class CollegeEnrichmentEngine:
    """
    Enhanced Federated Knowledge Discovery and Enrichment Engine for PragyanAI.
    Concurrently queries multi-engine search streams and normalizes unstructured 
    educational metrics using llama-3.3-70b-versatile with robust JSON boundary hooks.
    """
    def __init__(self):
        # 1. Primary Academic Knowledge Base: Wikipedia API Wrapper
        self.wiki = WikipediaAPIWrapper()
        
        # 2. Premium Live Web Indexer: Google Search via SerpAPI
        self.serp_search = SerpAPIWrapper()
        
        # 3. Fallback Open-Web Indexer: DuckDuckGo Search (Prevents credit depletion)
        self.ddg_search = DuckDuckGoSearchRun()
        
        # 4. Upgraded LLM Core Processor: Low-temperature high-capacity model
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.0)

    def _query_google_stream(self, query: str) -> str:
        """Isolated worker thread for premium Google search tracking."""
        try:
            return f"=== GOOGLE SEARCH STREAM RES ===\n{self.serp_search.run(query)}"
        except Exception as e:
            return f"Google Search Stream Bypassed/Failed: {str(e)}"

    def _query_ddg_stream(self, query: str) -> str:
        """Isolated worker thread for fallback DuckDuckGo search tracking."""
        try:
            return f"=== DUCKDUCKGO STREAM RES ===\n{self.ddg_search.run(query)}"
        except Exception as e:
            return f"DuckDuckGo Stream Bypassed/Failed: {str(e)}"

    def _query_wikipedia_stream(self, college_name: str, city: str) -> str:
        """Isolated worker thread for encyclopedic Wikipedia search tracking."""
        try:
            return f"=== WIKIPEDIA ARTICLE STREAM RES ===\n{self.wiki.run(f'{college_name} {city}')}"
        except Exception as e:
            return f"Wikipedia Stream Bypassed/Failed: {str(e)}"

    def discover_college_details(self, college_name: str, city: str) -> dict:
        """
        Orchestrates parallel multi-threaded federated lookups for a target institution.
        Normalizes unstructured textual contexts safely into database relational rows.
        """
        search_query = f"{college_name} {city} official website NAAC Grade NBA Accreditation NIRF Ranking"
        context_accumulator = []

        # --- PHASE 1: ASYNCHRONOUS CONCURRENT EXECUTIONS ---
        # Fires all three lookup streams concurrently to eliminate multi-second linear overhead lags
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_google = executor.submit(self._query_google_stream, search_query)
            future_ddg = executor.submit(self._query_ddg_stream, search_query)
            future_wiki = executor.submit(self._query_wikipedia_stream, college_name, city)

            # Wait and collect outputs from all completed worker threads
            context_accumulator.append(future_google.result())
            context_accumulator.append(future_ddg.result())
            context_accumulator.append(future_wiki.result())

        compiled_search_context = "\n\n".join(context_accumulator)

        # --- PHASE 2: GEN-AI SEMANTIC EXTRACTION SYSTEM ---
        template = """
        You are an elite educational intelligence agent parsing web search context logs for administrative records.
        Analyze the provided context streams thoroughly and isolate official verification parameters for: {target_college} located in {target_city}.
        
        Context Input Streams:
        {raw_context}
        
        Extract the following parameters strictly based on facts present inside the text:
        1. Website URL (Provide ONLY the pure string URL path, e.g., "[https://www.bmsit.ac.in](https://www.bmsit.ac.in)". Do NOT use markdown link formatting like brackets or parentheses).
        2. NAAC Rating (Must be formatted as a standard uppercase grade, e.g., A++, A+, A, B, Not Accredited, or N/A).
        3. NBA Accreditation status (Specify cleanly: Yes, No, or Partially Accredited).
        4. NIRF Ranking (Provide the exact current numerical integer placement or ranking band range like "151-200". If not found, output "N/A").
        5. Summary Profile (A brief, authoritative 2-sentence structural summary detailing history and focus).
        
        Return the metrics strictly formatted inside a single valid JSON array block.
        Do not include any preamble, conversational commentary, or trailing text notes. Ensure all json properties are properly quoted.
        
        Target JSON Format to Output:
        {{
            "website": "[https://www.example.edu.in](https://www.example.edu.in)",
            "naac_rating": "A+",
            "nba_accredited": "Partially Accredited",
            "nirf_ranking": "151-200",
            "summary": "Established in 1992, this institution is an autonomous engineering campus focusing on deep tech studies and industrial development."
        }}
        """

        try:
            formatted_prompt = template.format(
                target_college=college_name,
                target_city=city,
                raw_context=compiled_search_context
            )
            
            # Low-latency semantic consolidation query via Groq
            llm_output = self.llm.invoke(formatted_prompt).content
            clean_content = llm_output.strip()
            
            # --- PHASE 3: FAULT-TOLERANT ARRAY ISOLATION ---
            # Employs an exact regex bounds finder to isolate pure JSON fields even if formatting strings shift
            json_match = re.search(r'\{.*\}', clean_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Direct string clipping fallback
                if "```json" in clean_content:
                    json_str = clean_content.split("```json")[-1].split("```")[0].strip()
                else:
                    start_idx = clean_content.find("{")
                    end_idx = clean_content.rfind("}") + 1
                    json_str = clean_content[start_idx:end_idx] if start_idx != -1 else clean_content

            extracted_data = json.loads(json_str)

        except Exception as e:
            # Secure default fallback initialization if layout structures collapse
            print(f"[Enrichment LLM Warning] Core formatting exception recovered: {str(e)}")
            extracted_data = {
                "website": "N/A",
                "naac_rating": "N/A",
                "nba_accredited": "N/A",
                "nirf_ranking": "N/A",
                "summary": "Error encountered during multi-engine parsing synchronization hooks."
            }

        # Keep primary verification keys absolute for downstream database merges and updates
        extracted_data["college_name"] = college_name.strip().upper()
        extracted_data["city"] = city.strip().upper()
        
        return extracted_data
