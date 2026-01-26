import streamlit as st

# 1. Page configurations
st.set_page_config(page_title="TOF-MS Ringing Filter",
                   #page_title="Filter4Ringing",
                   page_icon="⚖️",
                   layout="wide",
                   initial_sidebar_state="expanded",
                   )

# 2. Define your pages 
# You can point to the files where your actual code lives
home_page = st.Page("pages/0_home.py", title="Home", icon="🏠", default=True)
demo_page = st.Page("pages/1_Interactive_Demo.py", title="Interactive Demo", icon="💡")
data_page = st.Page("pages/2_Import_Filter_Data.py", title="Import & Filter Data", icon="⚙️")

# 3. Create the navigation structure
pg = st.navigation([home_page, demo_page, data_page])
# 4. Run the navigation
pg.run()