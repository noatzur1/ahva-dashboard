import streamlit as st
import pandas as pd

# Basic page config
st.set_page_config(page_title="Ahva Test", layout="wide")

# Large, visible title
st.markdown("# 🎯 AHVA DASHBOARD TEST")
st.markdown("## If you see this, the app is working!")

# Big colored boxes to make sure something is visible
st.success("✅ SUCCESS: App is loading correctly!")
st.info("ℹ️ INFO: This is a test version")
st.warning("⚠️ WARNING: Upload your Excel file below")
st.error("❌ ERROR: This is just a test message")

# Simple file uploader
st.markdown("---")
st.markdown("## 📁 File Upload Test")

uploaded_file = st.file_uploader(
    "🔄 Choose your Excel file here:",
    type=['xlsx', 'xls'],
    help="Select your Ahva data file"
)

if uploaded_file:
    st.balloons()  # Fun animation
    st.success("🎉 File uploaded successfully!")
    
    try:
        df = pd.read_excel(uploaded_file)
        st.write(f"📊 **File loaded:** {len(df)} rows, {len(df.columns)} columns")
        st.write("**First 5 rows:**")
        st.dataframe(df.head())
        
        # Basic stats
        st.write("**Column names:**")
        for i, col in enumerate(df.columns, 1):
            st.write(f"{i}. {col}")
            
    except Exception as e:
        st.error(f"Error reading file: {e}")

# Sidebar test
st.sidebar.markdown("# 🧭 Sidebar Test")
st.sidebar.success("Sidebar is working!")

test_option = st.sidebar.selectbox(
    "Choose test option:",
    ["Option 1", "Option 2", "Option 3"]
)

st.sidebar.write(f"You selected: {test_option}")

# Footer
st.markdown("---")
st.markdown("### 🔧 Debug Information")
st.write("- Streamlit version: Working")
st.write("- Pandas: Working") 
st.write("- Page status: ✅ Loaded")

# Large end message
st.markdown("---")
st.markdown("# 🎯 END OF TEST PAGE")
st.markdown("## If you see this message, everything is working correctly!")
