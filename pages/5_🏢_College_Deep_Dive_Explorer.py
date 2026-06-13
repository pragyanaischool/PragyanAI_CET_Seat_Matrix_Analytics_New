import streamlit as st
import pandas as pd
from database.db_handler import get_combined_analytics

def render_deep_dive_explorer():
    """
    Renders the Comprehensive Institutional Explorer Center panel, 
    supporting dynamic pivot data mapping and multi-year trend highlights.
    """
    st.set_page_config(
        page_title="College Deep Dive Explorer",
        page_icon="🏢",
        layout="wide"
    )

    # 1. Page Header Branding UI Block
    st.title("🏢 Comprehensive Institutional Explorer Center")
    st.subheader("Isolate specific college profiles to evaluate baseline metrics, official URLs, and cross-year changes.")
    st.markdown("---")

    # Fetch combined dataset containing matrix parameters and cached web-scraped credentials
    df_lake = get_combined_analytics()

    # 2. Workspace Flow Boundary Verification Check
    if df_lake.empty:
        st.info("💡 Central database structures are currently unpopulated. Complete initial data extraction under Page 1 first.")
        return

    st.markdown("### 🏛️ Select Target Institution")
    
    # Extract unique institutional names present across the records to feed your dropdown widget
    available_colleges = sorted(df_lake['college_name'].unique())
    selected_college = st.selectbox(
        "Choose an institution to inspect:",
        options=available_colleges,
        help="Select the exact college entity you want to audit for multi-year program intake spreads."
    )

    # Filter full database data frame context down to this selected university target row subset
    college_subset_df = df_lake[df_lake['college_name'] == selected_college]
    
    # Extract parent-level metadata identifiers securely from the first available row index reference
    sample_profile_row = college_subset_df.iloc[0]

    st.markdown("---")

    # 3. EXECUTIVE FACT SHEET DATA METRICS LAYOUT
    st.markdown(f"## 📊 Executive Summary Fact Sheet: {selected_college}")
    
    # High-level regional and open-web verification metrics cards grid
    meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
    with meta_c1:
        st.metric(label="City Location Node", value=str(sample_profile_row.get('city', 'N/A')))
    with meta_c2:
        st.metric(label="Geographic District Area", value=str(sample_profile_row.get('district', 'N/A')))
    with meta_c3:
        st.metric(label="Current NAAC Standing", value=str(sample_profile_row.get('naac_rating', 'N/A')))
    with meta_c4:
        st.metric(label="NIRF Ranking Index", value=str(sample_profile_row.get('nirf_ranking', 'N/A')))

    st.markdown("---")

    # 4. LOCATION ADDRESS & EXECUTIVE ABSTRACT SUMMARY PRESENTATION LAYOUT
    layout_col_left, layout_col_right = st.columns([1, 2])

    with layout_col_left:
        st.markdown("#### 📍 Physical Verification Address Location")
        # Format physical address cleanly using code layout styles
        st.code(
            sample_profile_row.get('address', 'Official location address data block un-extracted.'), 
            language="text"
        )
        
        # Display official portal domain link as a clickable markdown hyperlink element if present in cache
        web_domain = sample_profile_row.get('website')
        if web_domain and web_domain != 'N/A':
            st.markdown(f"🔗 **Official Institutional Web Link Portal:** [{web_domain}]({web_domain})")
        else:
            st.caption("⚠️ No verified website URL found in memory. Execute cross-validation loops inside Page 2.")

    with layout_col_right:
        st.markdown("#### ℹ️ Institutional Profile Abstract Summary")
        # Retrieve Llama3-generated summary vectors from Page 2 open-web scraping logs
        summary_text = sample_profile_row.get('summary')
        if pd.isna(summary_text) or summary_text == 'None' or not str(summary_text).strip():
            st.info("💡 Open-web knowledge graph parsing is pending for this target campus entity. Run Enrichment routines inside Page 2 to populate this description framework.")
        else:
            st.info(summary_text)

    st.markdown("---")

    # 5. DYNAMIC MULTI-YEAR PROGRAM INTAKE PIVOT DATATABLE SCOPE
    st.markdown("### 💻 Program Capacity Intake Allocation Spread Matrix")
    st.markdown("Review the total allotted seats categorized across registered engineering tracks over multiple years:")

    try:
        # Construct cross-tabular pivot grid turning disjoint rows into historical tracking frames side-by-side
        # 'dept' becomes index labels, 'intake_year' expands into independent columns, cell values fill with seat capacities
        pivot_intake_matrix = college_subset_df.pivot_table(
            index='dept',
            columns='intake_year',
            values='intake',
            aggfunc='sum'
        ).fillna(0).astype(int)

        # Render clean visual data table with explicit column sorting configurations
        st.dataframe(
            pivot_intake_matrix,
            use_container_width=True,
            column_config={
                "dept": "Registered Academic Engineering Discipline Track",
                # Year integers inside column configs are customized automatically to block decimal displays
                **{year: st.column_config.NumberColumn(f"Intake Year {year}", format="%d") for year in pivot_intake_matrix.columns}
            }
        )

        # 6. AUTOMATED TELEMETRY INTENT LINE STATISTICS HIGHLIGHTS
        st.markdown("#### 📉 Analytical Growth Trend Insights")
        
        # Sequentially loop through our pivoted timeline columns to compute total seat sums for every financial cycle
        for cycle_year in sorted(pivot_intake_matrix.columns):
            total_seats_for_year = pivot_intake_matrix[cycle_year].sum()
            st.write(f"• Total Combined **{cycle_year}** Engineering Academic Intake Capacity: **{total_seats_for_year:,}** active allotted seats across branches.")

    except Exception as pivot_err:
        st.error(f"Failed to compile cross-year trend visualization tables: {str(pivot_err)}")
        st.warning("Please verify that multiple independent calendar year documents have been ingested via Page 1 to enable trend tracking workflows.")

if __name__ == "__main__":
    render_deep_dive_explorer()
  
