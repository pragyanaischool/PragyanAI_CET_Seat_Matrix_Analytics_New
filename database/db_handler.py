import os
import sqlite3
import pandas as pd

# Resolve the project path dynamically to maintain thread-safe integrity across multi-page views
DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database"))
DB_PATH = os.path.join(DB_DIR, "seat_matrix.db")

def init_database():
    """
    Initializes the local SQLite relational database schema with speed-optimized index anchors.
    Explicitly creates isolated schemas for multi-year seat matrices and external enrichment.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Unified Year-by-Year Master Seat Matrix Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS seat_matrix (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            college_name TEXT NOT NULL,
            city TEXT,
            district TEXT,
            address TEXT,
            dept TEXT,
            intake INTEGER,
            intake_year INTEGER
        )
    """)
    
    # 2. Open-Web Enrichment & Accreditation Cache Table
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
    
    # 🎯 HIGH-VELOCITY COVERING INDEXES (PREVENTS WORKSPACE TIMEOUTS DURING AGGREGATIONS)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_horizon_lookup ON seat_matrix(intake_year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_composite_join ON seat_matrix(college_name);")
    
    conn.commit()
    conn.close()

def save_matrix_records(df: pd.DataFrame):
    """
    Commits structured and normalized data frame entries into the master seat_matrix table.
    Automatically catches empty or malformed inputs to safeguard database integrity.
    
    Parameters:
        df (pd.DataFrame): Normalized multi-engine extracted records payload block.
    """
    if df is None or df.empty:
        print("[Database Log Warning] Blocked an attempt to write an empty or unallocated DataFrame.")
        return
        
    init_database()
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Enforce strict uniform types before execution loops to eliminate data type mismatches
        write_df = df.copy()
        
        # Keep tracking columns matching core extraction parameters
        core_columns = ['college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year']
        existing_cols = [col for col in core_columns if col in write_df.columns]
        write_df = write_df[existing_cols]
        
        if 'intake' in write_df.columns:
            write_df['intake'] = pd.to_numeric(write_df['intake'], errors='coerce').fillna(0).astype(int)
        if 'intake_year' in write_df.columns:
            write_df['intake_year'] = pd.to_numeric(write_df['intake_year'], errors='coerce').fillna(2024).astype(int)
            
        # Standardize strings to upper-case layout bounds to prevent case duplication anomalies
        for col in ['college_name', 'city', 'district', 'dept']:
            if col in write_df.columns:
                write_df[col] = write_df[col].astype(str).str.strip().str.upper()

        # Append rows transactionally into the local database table
        write_df.to_sql("seat_matrix", conn, if_exists="append", index=False)
    except Exception as tx_fault:
        print(f"[Database Critical Error] Matrix writing execution dropped: {str(tx_fault)}")
        raise tx_fault
    finally:
        conn.close()

def save_enrichment_record(record: dict):
    """
    Inserts or overwrites external web parameters for a specific college entity.
    Forces uniform uppercase structural keys to eliminate relational matching gaps.
    
    Parameters:
        record (dict): Clean JSON-derived dictionary mapping enrichment parameters.
    """
    if not record or not record.get('college_name'):
        return
        
    init_database()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Secure uniform upper-casing on primary index tracking hooks
    target_college_clean = str(record.get('college_name')).strip().upper()
    
    try:
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
            target_college_clean,
            str(record.get('website', 'N/A')).strip(),
            str(record.get('naac_rating', 'N/A')).strip().upper(),
            str(record.get('nba_accredited', 'N/A')).strip(),
            str(record.get('nirf_ranking', 'N/A')).strip().upper(),
            str(record.get('summary', 'No summary profile available.')).strip()
        ))
        conn.commit()
    except Exception as tx_fault:
        print(f"[Database Critical Error] Enrichment entry writing dropped: {str(tx_fault)}")
    finally:
        conn.close()

def get_combined_analytics() -> pd.DataFrame:
    """
    Queries and extracts full relational data tables via optimized left joins.
    Ensures safe alignment across key strings to prevent data drops on frontend visuals.
    
    Returns:
        pd.DataFrame: Clean data frame with uniform types to feed analytics pipelines.
    """
    init_database()
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(columns=['id', 'college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year', 'website', 'naac_rating', 'nba_accredited', 'nirf_ranking', 'summary'])
        
    conn = sqlite3.connect(DB_PATH)
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
            coalesce(ce.website, 'N/A') as website,
            coalesce(ce.naac_rating, 'N/A') as naac_rating,
            coalesce(ce.nba_accredited, 'N/A') as nba_accredited,
            coalesce(ce.nirf_ranking, 'N/A') as nirf_ranking,
            coalesce(ce.summary, 'No verification data available.') as summary
        FROM seat_matrix sm
        LEFT JOIN college_enrichment ce ON sm.college_name = ce.college_name
    """
    try:
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            # Enforce pristine types to shield downstream calculation blocks
            df['intake_year'] = pd.to_numeric(df['intake_year']).astype(int)
            df['intake'] = pd.to_numeric(df['intake']).astype(int)
            df['college_name'] = df['college_name'].astype(str).str.strip().str.upper()
            df['dept'] = df['dept'].astype(str).str.strip().str.upper()
            df['district'] = df['district'].astype(str).str.strip().str.upper()
            return df
    except Exception as query_fault:
        print(f"[Database Query Failure] Bypassed analytical stream readout: {str(query_fault)}")
    finally:
        conn.close()
        
    return pd.DataFrame(columns=['id', 'college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year', 'website', 'naac_rating', 'nba_accredited', 'nirf_ranking', 'summary'])

def clear_all_data():
    """
    Safely flushes operational table structures when resetting data ingestion channels.
    """
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS seat_matrix")
    cursor.execute("DROP TABLE IF EXISTS college_enrichment")
    conn.commit()
    conn.close()
    init_database()
