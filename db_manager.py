# db_manager.py
import sqlite3
import pandas as pd
import os

DB_NAME = "matrix_records.db"

def get_connection():
    """Establishes and returns a robust connection to the SQLite local database file."""
    return sqlite3.connect(DB_NAME)

def init_db():
    """
    Initializes the relational database schema if it doesn't already exist.
    Creates a unified schema designed to support both raw layout data extractions 
    and secondary asynchronous web-scraped metrics.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS colleges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college_name TEXT,
            city TEXT,
            district TEXT,
            address TEXT,
            dept TEXT,
            intake INTEGER,
            intake_year INTEGER,
            website TEXT DEFAULT 'Pending Lookup',
            naac TEXT DEFAULT 'Pending Lookup',
            nba TEXT DEFAULT 'Pending Lookup',
            nirf_rank TEXT DEFAULT 'Pending Lookup'
        )
    """)
    conn.commit()
    conn.close()

def save_chunk_to_sqlite(df: pd.DataFrame) -> int:
    """
    Appends a structured DataFrame chunk segment directly into the colleges table.
    Any property values that are missing or set to NaN drop gracefully into SQL NULL values.
    
    Args:
        df (pd.DataFrame): The slice or block of rows extracted via a Groq LLM pipeline chunk.
        
    Returns:
        int: Total number of rows currently residing in the database after the transaction.
    """
    conn = get_connection()
    
    # We enforce 'append' mode to ensure part-by-part stream dumps assemble incrementally
    df.to_sql("colleges", conn, if_exists="append", index=False)
    conn.commit()
    
    # Run immediate validation query check to track database row counter modifications
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM colleges")
    total_rows = cursor.fetchone()[0]
    conn.close()
    return total_rows

def load_full_dataset() -> pd.DataFrame:
    """
    Reads and returns the full synchronized content profile from the colleges table.
    
    Returns:
        pd.DataFrame: A unified DataFrame. Returns an empty DataFrame if no records are populated.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM colleges", conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def update_enriched_records(df: pd.DataFrame):
    """
    Overwrites the underlying database file records completely with a fully 
    aggregated and enriched metadata frame. This maps web scrapers' discoveries 
    directly over initial 'Pending Lookup' placeholders.
    
    Args:
        df (pd.DataFrame): Consolidated master dataset containing updated accreditation cells.
    """
    conn = get_connection()
    df.to_sql("colleges", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def run_filtered_query(districts: list, branches: list, intake_range: tuple) -> pd.DataFrame:
    """
    Executes a highly customized parameterized SQL query condition rule block based on 
    active dashboard filters. This logic executes directly at the database engine level 
    for maximum compute performance.
    
    Args:
        districts (list): Selected locations (e.g., ['BANGALORE URBAN', 'BALLARI'])
        branches (list): Target academic branches (e.g., ['COMPUTER SCIENCE AND ENGG'])
        intake_range (tuple): Slider constraint bounds (min_intake, max_intake)
        
    Returns:
        pd.DataFrame: Sliced and structured rows matching your target operational filters.
    """
    if not districts or not branches:
        return pd.DataFrame()
        
    conn = get_connection()
    
    # Generate dynamic parameterized query arguments to safeguard database processing runs
    district_placeholders = ",".join(["?"] * len(districts))
    branch_placeholders = ",".join(["?"] * len(branches))
    
    query = f"""
        SELECT * FROM colleges 
        WHERE district IN ({district_placeholders})
          AND dept IN ({branch_placeholders})
          AND intake BETWEEN ? AND ?
    """
    
    # Unpack filter lists sequentially to pass as arguments
    query_parameters = list(districts) + list(branches) + [intake_range[0], intake_range[1]]
    
    try:
        df = pd.read_sql_query(query, conn, params=query_parameters)
    except Exception:
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

# Initialize relational database schemas automatically upon library module reference import
init_db()
