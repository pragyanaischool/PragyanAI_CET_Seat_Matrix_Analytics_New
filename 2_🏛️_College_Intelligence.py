import streamlit as st
import os
import sys
import sqlite3
import pandas as pd

# ==============================================================================
# 🎯 PATH PATCH & CACHE PROTECTION LAYER (PREVENTS INTERMITTENT KEYERRORS)
# ==============================================================================
# Resolve absolute root bounds to guarantee module accessibility across sub-pages
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# Flush old cached module fingerprints out of memory during active session shifting
for module_key in list(sys.modules.keys()):
    if (
        module_key.startswith("engines") 
        or module_key.startswith("database") 
        or "matrix_parser" in module_key
        or "ensemble_orchestrator" in module_key
        or "enrichment_engine" in module_key
    ):
        sys.modules.pop(module_key, None)
# ==============================================================================

def load_intelligence_data():
    """
    Queries and extracts full relational data tables from the secure SQLite data lake.
    
    Returns:
        pd.DataFrame: Sanitized historical matrix data workspace.
    """
    try:
        db_path = os.path.join(root_path, "database", "seat_matrix.db")
        if not os.path.exists(db_path):
            return pd.DataFrame(columns=['college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year'])
            
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM seat_matrix_records", conn)
        conn.close()
        return df
    except Exception as e:
        print(f"[Database Connection Error] Failed to read analytical streams: {str(e)}")
        return pd.DataFrame(columns=['college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year'])

def render_college_intelligence():
    """
    Main Regional Intelligence View Canvas for PragyanAI.
    Renders high-level regional distributions, metrics cards, and district-level
    capacity footprints to evaluate engineering education trends in Karnataka.
    """
    st.set_page_config(
        page_title="PragyanAI College Intelligence", 
        page_icon="🏛️", 
        layout="wide"
    )
    
    st.title("🏛️ Regional College Intelligence Board")
    st.subheader("Aggregated multi-engine analytics mapping institutional density and seat capacities across districts.")
    st.markdown("---")

    # Load records from local analytical storage layer
    df = load_intelligence_data()

    if df.empty:
        st.warning("⚠️ Access Log Alert: Database data lake is empty. Please complete processing on page '1 Ingestion Portal' first.")
        st.info("💡 Next Step: Go to the 'Extraction & Ingestion' page in the sidebar and upload the official seat matrix document.")
        return

    # --- TOP CONTROL HUB & SLICERS ---
    st.sidebar.header("Intelligence Controls")
    
    # Dynamically extract available academic horizons compiled by ingestion runs
    available_years = sorted(df['intake_year'].unique().tolist(), reverse=True)
    selected_year = st.sidebar.selectbox("Select Academic Matrix Horizon", available_years)
    
    # Filter working data view based on selected horizontal index
    year_df = df[df['intake_year'] == selected_year]

    # --- SECTION 1: MACRO KPIS OVERVIEW PANELS ---
    st.markdown("### 📊 Macro Horizon Performance Indicators")
    col_inst, col_dept, col_seats = st.columns(3)
    
    with col_inst:
        total_colleges = year_df['college_name'].nunique()
        st.metric(
            label="Total Extracted Active Institutions", 
            value=total_colleges,
            help="Total count of unique engineering colleges parsed and validated across selected horizons."
        )
        
    with col_dept:
        total_depts = year_df['dept'].nunique()
        st.metric(
            label="Total Registered Academic Disciplines", 
            value=total_depts,
            help="Total count of unique course disciplines (branches) offered across the state."
        )
        
    with col_seats:
        total_intake = year_df['intake'].sum()
        st.metric(
            label="Aggregate Distributed State Intake", 
            value=f"{total_intake:,} Seats",
            help="Sum total of government and engineering division seats cataloged in this matrix dataset."
        )

    st.markdown("---")

    # --- SECTION 2: REGIONAL ANALYTICAL FOOTPRINTS ---
    st.markdown("### 📍 Geographic Capillary and Capacity Analysis")
    
    col_table, col_chart = st.columns([2, 3])
    
    # Aggregate data using the pre-sanitized district variables
    district_summary = (
        year_df.groupby('district')['intake']
        .sum()
        .reset_index()
        .sort_values(by='intake', ascending=False)
        .reset_index(drop=True)
    )
    
    # Add institutional volume count tracking columns dynamically
    district_colleges = year_df.groupby('district')['college_name'].nunique().reset_index()
    district_summary = pd.merge(district_summary, district_colleges, on='district')
    district_summary.columns = ['District Name', 'Total Allocated Intake Seats', 'Active Institutional Count']

    with col_table:
        st.markdown("#### 📋 District Allocation Breakdown Matrix")
        st.markdown("Sorted in descending order by aggregate seat capacity. Spelling variances are automatically unified via the cleaning rails:")
        st.dataframe(
            district_summary,
            use_container_width=True,
            hide_index=True
        )

    with col_chart:
        st.markdown("#### 📊 Regional Capacity Capacity Comparison Chart")
        st.markdown("Visual distribution footprint mapping seat opportunities per technical division hub:")
        # Render clean vertical comparative visual charts
        st.bar_chart(
            data=district_summary,
            x='District Name',
            y='Total Allocated Intake Seats',
            use_container_width=True
        )

    # --- SECTION 3: COURSE LEVEL DISTRIBUTION MARGINS ---
    st.markdown("---")
    st.markdown("### 🚀 Discipline Footprint Distribution")
    st.markdown("Top 15 academic disciplines ordered by total across-the-state seat allocations:")
    
    course_summary = (
        year_df.groupby('dept')['intake']
        .sum()
        .reset_index()
        .sort_values(by='intake', ascending=False)
        .head(15)
        .reset_index(drop=True)
    )
    course_summary.columns = ['Engineering Branch Discipline', 'Aggregated Allocated Seats']
    
    st.bar_chart(
        data=course_summary,
        x='Engineering Branch Discipline',
        y='Aggregated Allocated Seats',
        use_container_width=True
    )

if __name__ == "__main__":
    render_college_intelligence()
