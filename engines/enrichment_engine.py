import json
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

class CollegeEnrichmentEngine:
    """
    A federated knowledge-graph discovery engine for institutional data enrichment.
    Concurrently queries Google Search (SerpAPI), DuckDuckGo, and Wikipedia wrappers 
    to extract official URLs, NAAC ratings, NBA accreditation metrics, and NIRF rankings.
    """
    def __init__(self):
        # 1. Primary Academic Knowledge Base: Wikipedia API Wrapper
        self.wiki = WikipediaAPIWrapper()
        
        # 2. Premium Live Web Indexer: Google Search via SerpAPI
        self.serp_search = SerpAPIWrapper()
        
        # 3. Fallback Open-Web Indexer: DuckDuckGo Search (Prevents credit depletion/rate blocks)
        self.ddg_search = DuckDuckGoSearchRun()
        
        # 4. LLM Processor: Low-temperature Llama 3 (8B) to structure unstructured search contexts
        self.llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.1)

    def discover_college_details(self, college_name: str, city: str) -> dict:
        """
        Orchestrates federated lookups for a specific target campus profile.
        
        Parameters:
            college_name (str): The official extracted name of the college.
            city (str): The municipal zone where the campus is located.
            
        Returns:
            dict: Structured institutional parameters ready for SQLite cache storage.
        """
        # Formulate a precise query ensuring localized parameters
        search_query = f"{college_name} {city} official website NAAC Grade NBA Accreditation NIRF Ranking"
        context_accumulator = []
        
        # --- PHASE 1: FEDERATED OPEN-WEB SCRAPING DECK ---
        
        # Stream 1: Gather Google Search results
        try:
            google_res = self.serp_search.run(search_query)
            context_accumulator.append(f"=== GOOGLE SEARCH STREAM RES ===\n{google_res}")
        except Exception as e:
            context_accumulator.append(f"Google Search Stream Bypassed/Failed: {str(e)}")
            
        # Stream 2: Gather DuckDuckGo results (Acts as immediate failover protection)
        try:
            ddg_res = self.ddg_search.run(search_query)
            context_accumulator.append(f"=== DUCKDUCKGO STREAM RES ===\n{ddg_res}")
        except Exception as e:
            context_accumulator.append(f"DuckDuckGo Stream Bypassed/Failed: {str(e)}")
            
        # Stream 3: Gather Encyclopedic data from Wikipedia
        try:
            wiki_res = self.wiki.run(f"{college_name} {city}")
            context_accumulator.append(f"=== WIKIPEDIA ARTICLE STREAM RES ===\n{wiki_res}")
        except Exception as e:
            context_accumulator.append(f"Wikipedia Stream Bypassed/Failed: {str(e)}")

        # Consolidate all captured context layers into a unified block
        compiled_search_context = "\n\n".join(context_accumulator)

        # --- PHASE 2: GEN-AI SEMANTIC EXTRACTION DECK ---
        
        template = """
        You are an elite educational intelligence agent parsing web search context logs for administrative records.
        Analyze the provided context streams thoroughly and isolate official verification parameters for: {target_college} located in {target_city}.
        
        Context Input Streams:
        {raw_context}
        
        Extract the following parameters based on facts present inside the text:
        1. Official Domain Website URL
        2. NAAC Rating (e.g., A++, A, B, Not Accredited, N/A)
        3. NBA Accreditation status (e.g., Yes, No, Partially Accredited for specific courses)
        4. NIRF Ranking (Provide specific current numerical placement integer or ranking band range like 151-200. If missing, specify N/A)
        5. Summary Profile (A brief, authoritative 2-sentence structural summary detailing history and focus)
        
        You must return the metrics strictly formatted inside a single valid JSON markdown codeblock. 
        Do not add any preamble, conversational commentary, or trailing text notes. Ensure all json properties are properly quoted.
        
        JSON Structure Format:
        ```json
        {{
            "website": "[https://www.example.edu.in](https://www.example.edu.in)",
            "naac_rating": "A+",
            "nba_accredited": "Partially",
            "nirf_ranking": "151-200",
            "summary": "Established in 1992, this institution is an autonomous engineering campus focusing on deep tech studies and industrial development."
        }}
        ```
        """
        
        try:
            formatted_prompt = template.format(
                target_college=college_name,
                target_city=city,
                raw_context=compiled_search_context
            )
            
            # Execute low-latency structural generation on Groq frames
            llm_output = self.llm.invoke(formatted_prompt).content
            
            # Strip possible markdown delimiters from the output safely
            json_payload_str = llm_output.split("```json")[-1].split("```")[0].strip()
            extracted_data = json.loads(json_payload_str)
            
        except Exception:
            # Secure default fallback instantiation if formatting boundaries collapse
            extracted_data = {
                "website": "N/A",
                "naac_rating": "N/A",
                "nba_accredited": "N/A",
                "nirf_ranking": "N/A",
                "summary": "Error encountered during multi-engine parsing synchronization hooks."
            }
            
        # Ensure the structural primary link-key remains pinned for SQL Joins
        extracted_data["college_name"] = college_name
        return extracted_data
      
