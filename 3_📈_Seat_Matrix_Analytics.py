import streamlit as st
import os
import sys

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

import pandas as pd
import sqlite3

def get_db_connection():
    """Establishes a thread-safe connection to the unified relational database data lake."""
    db_path = os.path.join(root_path, "database", "seat_matrix.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_comprehensive_analytics():
    """
    Queries and normalizes integrated historical matrix data streams.
    Ensures safe alignment across structural types to prevent aggregation faults.
    """
    try:
        conn = get_db_connection()
        # Extract full historical records matrix
        query = "SELECT * FROM seat_matrix_records"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            # Enforce strict uniform types on primary keys to prevent comparison alignment drops
            df['intake_year'] = pd.to_numeric(df['intake_year']).astype(int)
            df['intake'] = pd.to_numeric(df['intake']).astype(int)
            df['college_name'] = df['college_name'].astype(str).str.strip().str.upper()
            df['dept'] = df['dept'].astype(str).str.strip().str.upper()
            df['district'] = df['district'].astype(str).str.strip().str.upper()
            return df
    except Exception as e:
        print(f"[Analytics Ingest Alert] Relational stream lookup bypassed: {str(e)}")
    
    # Return empty schema-compliant fallback block if transaction drops
    return pd.DataFrame(columns=['college_name', 'city', 'district', 'address', 'dept', 'intake', 'intake_year'])

def render_analytics_dashboard():
    """
    Main Multi-Dimensional Comparative Dashboard Panel for PragyanAI.
    Coordinates structural aggregations, regional visual charts, institutional 
    locators, and advanced Year-over-Year anomaly trajectory diagnostic matrix metrics.
    """
    st.set_page_config(
        page_title="PragyanAI Seat Analytics",
        page_icon="📈",
        layout="wide"
    )

    # 1. Page Header Identification Controls
    st.title("📈 Multi-Dimensional Seat Allocation Analytics Workspace")
    st.subheader("Execute structural aggregations, regional distributions, and year-over-year variance diagnostic loops.")
    st.markdown("---")

    # Fetch normalized integrated records from local relational database data lake
    master_df = load_comprehensive_analytics()

    if master_df.empty:
        st.warning("⚠️ Metrics Ingestion Log: Relational repository data lake is currently unpopulated.")
        st.info("💡 Next Step: Go to page '1 Extraction & Ingestion' in the sidebar panel and process your raw source document.")
        return

    # 2. Sidebar Filter Controls Setup
    st.sidebar.header("🎛️ Workspace Control Panel")
    
    # Sort years descending to present newest intake insights first
    available_years = sorted(master_df['intake_year'].unique().tolist(), reverse=True)
    selected_year = st.sidebar.selectbox(
        "Isolate Target Data Horizon Year:",
        options=available_years,
        index=0,
        help="Filters metrics displayed across tabs 1, 2, and 3 to a single structural academic timeline context."
    )

    # Filter base slice for current year tracking tabs
    active_yr_df = master_df[master_df['intake_year'] == selected_year]

    # 3. Core High-Level Metric Cards Panel
    st.markdown(f"### 🌍 State-Wide Metric Summary — Academic Session {selected_year}")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Total Allocated Seat Capacity", 
            value=f"{active_yr_df['intake'].sum():,} Seats"
        )
    with col_m2:
        st.metric(
            label="Unique Campuses Catalogued", 
            value=f"{active_yr_df['college_name'].nunique():,}"
        )
    with col_m3:
        st.metric(
            label="Active Academic Departments", 
            value=f"{active_yr_df['dept'].nunique():,}"
        )
    with col_m4:
        st.metric(
            label="Unique Geographic Districts", 
            value=f"{active_yr_df['district'].nunique():,}"
        )

    st.markdown("---")

    # 4. Multi-Tab Presentation Router Setup
    tab_districts, tab_departments, tab_finder, tab_yoy_deltas = st.tabs([
        "📍 District-Wise Analysis", 
        "💻 Department-Wise Analysis", 
        "🏛️ Specific District Finder", 
        "🔄 YoY Systems Variance"
    ])

    # --- TAB 1: DISTRICT-WISE DIAGNOSTICS ---
    with tab_districts:
        st.markdown(f"### 📍 Regional Seat Allocation and Distribution Density ({selected_year})")
        
        # Calculate structural district summaries
        district_summary = active_yr_df.groupby('district')['intake'].sum().reset_index()
        district_summary = district_summary.sort_values(by='intake', ascending=False).reset_index(drop=True)
        
        col_t1_l, col_t1_r = st.columns([2, 3])
        with col_t1_l:
            st.markdown("**Tabular District Matrix View**")
            st.dataframe(
                district_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "district": "Geographic District Node",
                    "intake": st.column_config.NumberColumn("Total Seats Capacity", format="%d")
                }
            )
        with col_t1_r:
            st.markdown("**Visual Spatial Intake Volume Chart**")
            st.bar_chart(
                data=district_summary,
                x="district",
                y="intake",
                use_container_width=True
            )

    # --- TAB 2: DEPARTMENT-WISE DIAGNOSTICS ---
    with tab_departments:
        st.markdown(f"### 💻 Technical Specialization Volume Analytics ({selected_year})")
        
        dept_summary = active_yr_df.groupby('dept')['intake'].sum().reset_index()
        dept_summary = dept_summary.sort_values(by='intake', ascending=False).reset_index(drop=True)
        
        # Isolate top 15 categories to prevent graph clutter on dashboard view ports
        top_15_depts = dept_summary.head(15)
        
        col_t2_l, col_t2_r = st.columns([3, 2])
        with col_t2_l:
            st.markdown("**Top 15 Technical Specialties Breakdown**")
            st.bar_chart(
                data=top_15_depts,
                x="dept",
                y="intake",
                use_container_width=True
            )
        with col_t2_r:
            st.markdown("**Complete Program Capacity Matrix**")
            st.dataframe(
                dept_summary,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "dept": "Academic Course/Track Label",
                    "intake": st.column_config.NumberColumn("Total Seat Volume", format="%d")
                }
            )

    # --- TAB 3: DISTRICT INSTITUTION LOCATOR FINDER ---
    with tab_finder:
        st.markdown(f"### 🏛️ Locate Campuses and Specializations Registered in Specific Districts ({selected_year})")
        
        available_districts = sorted(active_yr_df['district'].dropna().unique().tolist())
        chosen_district = st.selectbox(
            "Select Target District Area to Audit:",
            options=available_districts,
            help="Filters downstream listings exclusively to colleges physically located inside this geographic boundary."
        )
        
        filtered_district_df = active_yr_df[active_yr_df['district'] == chosen_district]
        st.markdown(f"Found **{filtered_district_df['college_name'].nunique()}** unique college institutes operating inside **{chosen_district}**:")
        
        display_columns = ['college_name', 'city', 'dept', 'intake', 'address']
        existing_cols = [c for c in display_columns if c in filtered_district_df.columns]
        
        st.dataframe(
            filtered_district_df[existing_cols].sort_values(by=['college_name', 'dept']).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            column_config={
                "college_name": "Institution Entity Name",
                "city": "City Node",
                "dept": "Course Specialty",
                "intake": "Intake Capacity",
                "address": "Physical Verification Address"
            }
        )

    # --- TAB 4: YEAR-OVER-YEAR DELTA MATRIX CALCULATOR ---
    with tab_yoy_deltas:
        st.markdown("### 🔄 Year-over-Year System Variance & Structural Tracking Anomaly Detector")
        
        historical_years = sorted(master_df['intake_year'].unique().tolist())
        
        if len(historical_years) < 2:
            st.info("ℹ️ Multi-Horizon Sequence Note: Year-over-Year structural comparison matrices unlock automatically once records for at least two separate years are ingested under Page 1.")
        else:
            col_delta1, col_delta2 = st.columns(2)
            with col_delta1:
                base_horizon = st.selectbox(
                    "Select Base Reference Horizon Year (T-1):",
                    options=historical_years,
                    index=0
                )
            with col_delta2:
                comp_horizon = st.selectbox(
                    "Select Comparison Target Horizon Year (T):",
                    options=historical_years,
                    index=min(1, len(historical_years) - 1)
                )
                
            if base_horizon == comp_horizon:
                st.warning("⚠️ Layout Validation Warning: Select two distinct calendar horizons to evaluate tracking growth trajectories.")
            else:
                # Extract clean string arrays to check for addition/deletion anomalies across timelines
                base_institutes = set(master_df[master_df['intake_year'] == base_horizon]['college_name'].dropna().unique().tolist())
                comp_institutes = set(master_df[master_df['intake_year'] == comp_horizon]['college_name'].dropna().unique().tolist())
                
                added_campuses = comp_institutes - base_institutes
                missing_campuses = base_institutes - comp_institutes
                
                st.markdown(f"#### 📊 Structural Comparison Diagnostics: {base_horizon} ➔ {comp_horizon}")
                
                col_anom_l, col_anom_r = st.columns(2)
                
                with col_anom_l:
                    st.success(f"➕ Newly Introduced Campus Entries in {comp_horizon}: **{len(added_campuses)}** Institutions")
                    if added_campuses:
                        st.dataframe(
                            pd.DataFrame(list(added_campuses), columns=["Newly Ingested Campus Entities"]),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No brand-new institutional campuses introduced across this structural timeline period.")
                        
                with col_anom_r:
                    st.error(f"❌ Dropped / Non-Reporting Campus Entries in {comp_horizon}: **{len(missing_campuses)}** Institutions")
                    if missing_campuses:
                        st.dataframe(
                            pd.DataFrame(list(missing_campuses), columns=["Dropped / Non-Reporting Campus Entities"]),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No baseline campuses dropped or omitted across this tracking scope sequence.")

    # 5. Universal Master Spreadsheet Export Terminal Link
    st.markdown("---")
    st.markdown("### 📥 Universal Integrated Intelligence Export Hub")
    st.markdown("Generate and stream out complete multi-year aggregated matrices containing parsed allocation counts.")
    
    try:
        csv_bytes_stream = master_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Unified Admissions Matrix File (CSV Format)",
            data=csv_bytes_stream,
            file_name="pragyan_ai_integrated_matrix_report.csv",
            mime="text/csv",
            use_container_width=True,
            help="Generates an offline spreadsheet file containing all combined seat matrices extracted across ingestion pipelines."
        )
    except Exception as export_err:
        st.error(f"Export download link serialization sequence failed: {str(export_err)}")

if __name__ == "__main__":
    render_analytics_dashboard()
