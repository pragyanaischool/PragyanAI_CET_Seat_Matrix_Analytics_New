import streamlit as st
import pandas as pd
from database.db_handler import get_combined_analytics

def render_analytics_dashboard():
    """
    Renders the multi-dimensional, comparative analytics dashboard panel,
    supporting district-wise, department-wise, and YoY anomaly diagnostics.
    """
    st.set_page_config(
        page_title="PragyanAI Seat Matrix Analytics",
        page_icon="📈",
        layout="wide"
    )

    # 1. Page Header Identification Header
    st.title("📈 Multi-Dimensional Seat Allocation Analytics Workspace")
    st.subheader("Execute structural aggregations, regional distributions, and year-over-year variance checks.")
    st.markdown("---")

    # Fetch fresh integrated dataset records from SQLite transactional queries
    master_df = get_combined_analytics()

    if master_df.empty:
        st.info("💡 Data metrics repository is currently unpopulated. Complete initial data extraction under Page 1 first.")
        return

    # 2. Sidebar Filter Controls Setup
    st.sidebar.header("🎛️ Workspace Control Panel")
    
    # Sort years descending to present newest intake insights first
    available_years = sorted(master_df['intake_year'].unique(), reverse=True)
    selected_year = st.sidebar.selectbox(
        "Isolate Target Data Horizon Year:",
        options=available_years,
        index=0,
        help="Filters the metrics displayed across tabs 1, 2, and 3 to a single financial cycle context."
    )

    # Filter base slice for current year tracking tabs
    active_yr_df = master_df[master_df['intake_year'] == selected_year]

    # 3. Core High-Level Metric Cards Panel
    st.markdown(f"### 🌍 State-Wide Metric Summary — Academic Session {selected_year}")
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric(
            label="Total Seat Capacity", 
            value=f"{active_yr_df['intake'].sum():,}"
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
        
        # Calculate aggregations
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
            st.markdown("**Visual Spacial Intake Volume Chart**")
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
        
        # Isolate top 15 categories to prevent graph clutter on dashboard
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
        
        available_districts = sorted(active_yr_df['district'].unique())
        chosen_district = st.selectbox(
            "Select Target District Area to Audit:",
            options=available_districts,
            help="Filters downstream listings exclusively to colleges physically located inside this geographic boundary."
        )
        
        filtered_district_df = active_yr_df[active_yr_df['district'] == chosen_district]
        
        st.markdown(f"Found **{filtered_district_df['college_name'].nunique()}** unique college institutes operating inside **{chosen_district}**:")
        
        display_columns = ['college_name', 'city', 'dept', 'intake', 'address', 'website']
        existing_cols = [c for c in display_columns if c in filtered_district_df.columns]
        
        st.dataframe(
            filtered_district_df[existing_cols].sort_values(by=['college_name', 'dept']),
            use_container_width=True,
            hide_index=True,
            column_config={
                "college_name": "Institution Entity Name",
                "city": "City Node",
                "dept": "Course Specialty",
                "intake": "Intake Capacity",
                "address": "Physical Verification Address",
                "website": st.column_config.LinkColumn("Official Web Link URL")
            }
        )

    # --- TAB 4: YEAR-OVER-YEAR DELTA MATRIX CALCULATOR ---
    with tab_yoy_deltas:
        st.markdown("### 🔄 Year-over-Year System Variance & Structural Tracking Anomaly Detector")
        
        historical_years = sorted(master_df['intake_year'].unique())
        
        if len(historical_years) < 2:
            st.info("ℹ️ Year-over-Year structural comparison matrix unlocks automatically once files for at least two separate years are ingested under Page 1.")
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
                st.warning("Select two distinct calendar horizons to evaluate structural trajectory growth curves.")
            else:
                # Isolate sets of unique entities operating inside respective year groups
                base_institutes = set(master_df[master_df['intake_year'] == base_horizon]['college_name'].unique())
                comp_institutes = set(master_df[master_df['intake_year'] == comp_horizon]['college_name'].unique())
                
                added_campuses = comp_institutes - base_institutes
                missing_campuses = base_institutes - comp_institutes
                
                st.markdown(f"#### 📊 Structural Comparison Diagnostics: {base_horizon} ➔ {comp_horizon}")
                
                col_anom_l, col_anom_r = st.columns(2)
                
                with col_anom_l:
                    st.success(f"➕ Newly Added Colleges in {comp_horizon}: **{len(added_campuses)}** Institutions")
                    if added_campuses:
                        st.dataframe(
                            pd.DataFrame(list(added_campuses), columns=["Newly Ingested Campus Entities"]),
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.write("No brand-new institutional campuses introduced across this structural timeline period.")
                        
                with col_anom_r:
                    st.error(f"❌ Closed / Missing Colleges in {comp_horizon} relative to {base_horizon}: **{len(missing_campuses)}** Institutions")
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
    st.markdown("Generate and stream out complete multi-year aggregated matrices containing parsed numbers and scraped rankings.")
    
    try:
        csv_bytes_stream = master_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Unified Admissions Matrix File (CSV Format)",
            data=csv_bytes_stream,
            file_name="pragyan_ai_integrated_matrix_report.csv",
            mime="text/csv",
            use_container_width=True,
            help="Generates an offline spreadsheet file containing all combined seat matrices and cached open-web validation credentials."
        )
    except Exception as export_err:
        st.error(f"Export download link serialization sequence failed: {str(export_err)}")

if __name__ == "__main__":
    render_analytics_dashboard()
