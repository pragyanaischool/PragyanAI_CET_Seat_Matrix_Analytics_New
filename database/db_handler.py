import sqlite3
import pandas as pd

DB_NAME = "pragyan_analytics.db"

def init_db():
    """
    Initializes the persistent SQLite database tables.
    Creates schemas for tracking multi-year seat allocation matrices
    and caching federated open-web institutional enrichment parameters.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Unified Year-by-Year Master Seat Matrix Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seat_matrix (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college_name TEXT,
            city TEXT,
            district TEXT,
            address TEXT,
            dept TEXT,
            intake INTEGER,
            intake_year INTEGER
        )
    """)
    
    # 2. Open-Web Enrichment & Accreditation Cache Table
    # college_name is a Primary Key to enforce a 1:Many relationship 
    # with seat_matrix rows (one college profile maps to multiple department seat rows)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS college_enrichment (
            college_name TEXT PRIMARY KEY,
            website TEXT,
            naac_rating TEXT,
            nba_accredited TEXT,
            nirf_ranking TEXT,
            summary TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def save_matrix_records(df: pd.DataFrame):
    """
    Appends freshly extracted document data frames into the master seat_matrix table.
    
    Parameters:
        df (pd.DataFrame): Normalized database rows containing parsed structural allocations.
    """
    conn = sqlite3.connect(DB_NAME)
    # Appends incoming logs to support multiple years simultaneously without collisions
    df.to_sql("seat_matrix", conn, if_exists="append", index=False)
    conn.close()

def save_enrichment_record(record: dict):
    """
    Inserts or overwrites external metrics (accreditations, rankings, domains) 
    for a specific collegiate institution entity.
    
    Parameters:
        record (dict): Clean JSON-derived dictionary mapping required intelligence properties.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO college_enrichment (
            college_name, 
            website, 
            naac_rating, 
            nba_accredited, 
            nirf_ranking, 
            summary
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        record.get('college_name'), 
        record.get('website', 'N/A'), 
        record.get('naac_rating', 'N/A'), 
        record.get('nba_accredited', 'N/A'), 
        record.get('nirf_ranking', 'N/A'), 
        record.get('summary', 'No summary profile available.')
    ))
    conn.commit()
    conn.close()

def get_combined_analytics() -> pd.DataFrame:
    """
    Executes a left join operation between your multi-year seat allocations 
    and open-web enrichment parameters to compile a unified database frame.
    
    Returns:
        pd.DataFrame: Complete tabular dataset framework matching downstream dashboard feeds.
    """
    conn = sqlite3.connect(DB_NAME)
    query = """
        SELECT 
            sm.id,
            sm.college_name,
            sm.city,
            sm.district,
            sm.address,
            sm.dept,
            sm.intake,
            sm.intake_year,
            ce.website,
            ce.naac_rating,
            ce.nba_accredited,
            ce.nirf_ranking,
            ce.summary
        FROM seat_matrix sm
        LEFT JOIN college_enrichment ce ON sm.college_name = ce.college_name
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def clear_all_data():
    """
    Utility function to flush database states when resetting system pipelines.
    """
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS seat_matrix")
    cursor.execute("DROP TABLE IF EXISTS college_enrichment")
    conn.commit()
    conn.close()
    init_db()
