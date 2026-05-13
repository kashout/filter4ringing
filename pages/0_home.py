import streamlit as st

st.title("Data-Driven Filter for Detector Oscillation Artifacts in Time-of-Flight Mass Spectrometry")

# Authors & Affiliations
st.markdown("""
Kas J. Houthuijs¹, Karl J. Jobst², Frederic Béen¹,³  
*¹Amsterdam Institute for Life and Environment (A-LIFE), Vrije Universiteit Amsterdam;
²Department of Chemistry, Memorial University of Newfoundland;
³KWR Water Research Institute.*
""")

st.divider()

st.markdown("""
Detector oscillations are intrinsic artifacts in time-of-flight mass spectrometry (TOF-MS) data and
can produce spurious signals that impede data analysis. This Streamlit application uses a data-driven
workflow to identify and remove these artifacts, thereby improving data quality.

The workflow relies on a five-stage approach:

1. **Strict grouping**: Identify feature pairs using strict co-elution, correlation and intensity criteria
2. **Maxima finding**: Apply Gaussian convolution to find recurring Δ√(m/z) values representing ringing artifacts
3. **Broad detection**: Use Δ√(m/z) values and broadened criteria to prevent fasle-negatives
4. **Criteria refinement**: Refine selection criteria to minimize false-positive annotations
5. **Refined filtering**: Re-processes data with refined parameters to accurately annotate and filter artifacts
""")

st.image("images/workflow_figure.png")

col1, col2, spacer = st.columns([1, 1, 3])
with col1:
    st.link_button("📄 Read more", "https://doi.org/10.1021/acs.analchem.6c00762")
with col2:
    with st.popover("📑 Cite Us"):
            st.markdown("""
> Houthuijs, K.J.; Jobst, K.J.; Béen, F. Data-Driven Filter for Detector Oscillation Artifacts in Time-of-Flight Mass Spectrometry. _Anal. Chem._ **2026**, XX, XX-XX. https://doi.org/10.1021/acs.analchem.6c00762""")

st.sidebar.success("Select a page above to begin.")

with st.sidebar:
    st.divider()
    st.subheader("Privacy & Security")
    st.info("""
    **Data Sovereignty:** This application is hosted on Streamlit Community Cloud. Uploaded files are processed 
    in-memory and are not stored permanently on their servers.
    """)
    
    st.warning("""
    **Terms of Service:**
    By uploading data, you acknowledge that this service is powered by Snowflake/Streamlit 
    and agree to their [Terms of Service](https://streamlit.io/terms-of-use). 
    
    *Do not upload sensitive, proprietary, or HIPAA-regulated data.*
    """)

# import pandas here becasue it can be slow
import pandas as pd
