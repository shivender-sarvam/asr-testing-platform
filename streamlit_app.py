import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
import io
import re

# Page config
st.set_page_config(
    page_title="ASR Testing Platform - DCS",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .crop-card {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .crop-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .recording-section {
        background: #fff;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .results-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_name' not in st.session_state:
    st.session_state.qa_name = ""
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = ""
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = ""
if 'crop_data' not in st.session_state:
    st.session_state.crop_data = []
if 'current_crops' not in st.session_state:
    st.session_state.current_crops = []
if 'selected_crops' not in st.session_state:
    st.session_state.selected_crops = []
if 'current_crop_index' not in st.session_state:
    st.session_state.current_crop_index = 0
if 'current_sentence_index' not in st.session_state:
    st.session_state.current_sentence_index = 0
if 'test_results' not in st.session_state:
    st.session_state.test_results = []
if 'testing_active' not in st.session_state:
    st.session_state.testing_active = False

# Default crop data
default_crops = {
    'hindi': [
        {'serial': 1, 'code': 'H001', 'name': 'पेठा', 'language': 'hindi', 'project': 'DCS'},
        {'serial': 2, 'code': 'H002', 'name': 'बैंगन', 'language': 'hindi', 'project': 'DCS'},
        {'serial': 3, 'code': 'H003', 'name': 'मक्का', 'language': 'hindi', 'project': 'DCS'},
        {'serial': 4, 'code': 'H004', 'name': 'गेहूं', 'language': 'hindi', 'project': 'DCS'},
        {'serial': 5, 'code': 'H005', 'name': 'चावल', 'language': 'hindi', 'project': 'DCS'}
    ],
    'english': [
        {'serial': 1, 'code': 'E001', 'name': 'Ash Gourd', 'language': 'english', 'project': 'DCS'},
        {'serial': 2, 'code': 'E002', 'name': 'Brinjal', 'language': 'english', 'project': 'DCS'},
        {'serial': 3, 'code': 'E003', 'name': 'Maize', 'language': 'english', 'project': 'DCS'},
        {'serial': 4, 'code': 'E004', 'name': 'Wheat', 'language': 'english', 'project': 'DCS'},
        {'serial': 5, 'code': 'E005', 'name': 'Rice', 'language': 'english', 'project': 'DCS'}
    ]
}

# Main header
st.markdown("""
<div class="main-header">
    <h1>🎤 ASR Testing Platform</h1>
    <h3>DCS Crop Name Testing</h3>
    <p>Test ASR accuracy for crop names in multiple languages</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.header("📋 Navigation")
    
    if not st.session_state.qa_name:
        st.info("👆 Enter your name to start")
    elif not st.session_state.selected_language:
        st.info("👆 Select language")
    elif not st.session_state.testing_active:
        st.info("👆 Upload CSV or start testing")
    else:
        st.success("✅ Testing in progress")
    
    st.markdown("---")
    st.markdown("### 📊 Current Status")
    if st.session_state.qa_name:
        st.write(f"**QA Name:** {st.session_state.qa_name}")
    if st.session_state.selected_language:
        st.write(f"**Language:** {st.session_state.selected_language.title()}")
    if st.session_state.selected_project:
        st.write(f"**Project:** {st.session_state.selected_project}")
    if st.session_state.current_crops:
        st.write(f"**Crops Available:** {len(st.session_state.current_crops)}")
    if st.session_state.testing_active:
        st.write(f"**Progress:** {st.session_state.current_crop_index + 1}/{len(st.session_state.selected_crops)}")

# Main content
if not st.session_state.qa_name:
    # Step 1: QA Name Input
    st.header("👤 Step 1: Enter Your Name")
    st.markdown("Enter your name to start the ASR testing process.")
    
    qa_name = st.text_input("QA Name", placeholder="Enter your name here...")
    
    if st.button("Continue", type="primary"):
        if qa_name.strip():
            st.session_state.qa_name = qa_name.strip()
            st.rerun()
        else:
            st.error("Please enter your name to continue.")

elif not st.session_state.selected_language:
    # Step 2: Language Selection
    st.header("🌍 Step 2: Select Language")
    st.markdown("Choose the language for ASR testing.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🇮🇳 Hindi", use_container_width=True):
            st.session_state.selected_language = "hindi"
            st.session_state.selected_project = "DCS"
            st.session_state.current_crops = default_crops['hindi']
            st.rerun()
    
    with col2:
        if st.button("🇺🇸 English", use_container_width=True):
            st.session_state.selected_language = "english"
            st.session_state.selected_project = "DCS"
            st.session_state.current_crops = default_crops['english']
            st.rerun()

elif not st.session_state.testing_active:
    # Step 3: CSV Upload or Start Testing
    st.header("📁 Step 3: Upload CSV or Start Testing")
    
    # CSV Upload Section
    st.subheader("📤 Upload CSV File")
    st.markdown("Upload a CSV file with crop data in the format: `serial_number,crop_code,crop_name,language,project`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Normalize column names
            df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
            
            # Check required columns
            required_columns = ['serial_number', 'crop_code', 'crop_name', 'language', 'project']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.write("Available columns:", list(df.columns))
            else:
                # Process CSV data
                crop_data = []
                for _, row in df.iterrows():
                    crop_data.append({
                        'serial': int(row['serial_number']),
                        'code': str(row['crop_code']),
                        'name': str(row['crop_name']),
                        'language': str(row['language']).lower(),
                        'project': str(row['project']).upper()
                    })
                
                # Filter by language and project
                filtered_crops = [crop for crop in crop_data 
                               if crop['language'] == st.session_state.selected_language 
                               and crop['project'] == st.session_state.selected_project]
                
                if filtered_crops:
                    st.session_state.crop_data = crop_data
                    st.session_state.current_crops = filtered_crops
                    st.success(f"✅ CSV uploaded successfully! Found {len(filtered_crops)} crops for {st.session_state.selected_language.title()} - {st.session_state.selected_project}")
                    
                    # Show preview
                    st.subheader("📋 Crop Preview")
                    preview_df = pd.DataFrame(filtered_crops[:5])  # Show first 5
                    st.dataframe(preview_df, use_container_width=True)
                    
                    if len(filtered_crops) > 5:
                        st.info(f"... and {len(filtered_crops) - 5} more crops")
                else:
                    st.warning(f"No crops found for {st.session_state.selected_language.title()} - {st.session_state.selected_project}")
                    
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
    
    # Start Testing Section
    st.subheader("🚀 Start Testing")
    
    if st.session_state.current_crops:
        st.info(f"Ready to test {len(st.session_state.current_crops)} crops")
        
        # Crop range selection
        col1, col2 = st.columns(2)
        with col1:
            start_range = st.number_input("Start from crop", min_value=1, max_value=len(st.session_state.current_crops), value=1)
        with col2:
            end_range = st.number_input("End at crop", min_value=start_range, max_value=len(st.session_state.current_crops), value=min(5, len(st.session_state.current_crops)))
        
        if st.button("🎤 Start ASR Testing", type="primary", use_container_width=True):
            # Select crops for testing
            selected_crops = st.session_state.current_crops[start_range-1:end_range]
            st.session_state.selected_crops = selected_crops
            st.session_state.current_crop_index = 0
            st.session_state.current_sentence_index = 0
            st.session_state.test_results = []
            st.session_state.testing_active = True
            st.rerun()
    else:
        st.warning("No crops available. Please upload a CSV file or use default crops.")

else:
    # Step 4: Testing Interface
    if st.session_state.current_crop_index < len(st.session_state.selected_crops):
        current_crop = st.session_state.selected_crops[st.session_state.current_crop_index]
        
        st.header(f"🎤 Testing: {current_crop['name']}")
        st.markdown(f"**Crop Code:** {current_crop['code']} | **Language:** {current_crop['language'].title()}")
        
        # Progress
        progress = (st.session_state.current_crop_index + 1) / len(st.session_state.selected_crops)
        st.progress(progress)
        st.caption(f"Progress: {st.session_state.current_crop_index + 1} of {len(st.session_state.selected_crops)} crops")
        
        # Current sentence
        sentences = [
            f"I am growing {current_crop['name']} in my field",
            f"My crop is {current_crop['name']} this season",
            f"I planted {current_crop['name']} last month",
            f"The {current_crop['name']} is growing well",
            f"I harvest {current_crop['name']} every year"
        ]
        
        current_sentence = sentences[st.session_state.current_sentence_index]
        
        st.subheader(f"📝 Sentence {st.session_state.current_sentence_index + 1}/5")
        st.markdown(f"**Say this sentence:** \"{current_sentence}\"")
        
        # Recording interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎤 Start Recording", type="primary"):
                st.info("Recording started... (This is a demo - actual recording would be implemented)")
        
        with col2:
            if st.button("⏹️ Stop Recording"):
                st.info("Recording stopped... (This is a demo - actual recording would be implemented)")
        
        with col3:
            if st.button("📤 Submit Recording"):
                # Mock ASR processing
                st.info("Processing audio with ASR... (This is a demo - actual ASR would be implemented)")
                
                # Mock keyword extraction
                extracted_keyword = mock_keyword_extraction(current_sentence, current_crop['name'])
                
                # Store result
                result = {
                    'crop_name': current_crop['name'],
                    'crop_code': current_crop['code'],
                    'sentence': current_sentence,
                    'extracted_keyword': extracted_keyword,
                    'is_correct': extracted_keyword.lower() == current_crop['name'].lower(),
                    'timestamp': datetime.now().isoformat()
                }
                
                st.session_state.test_results.append(result)
                
                # Move to next sentence or crop
                if st.session_state.current_sentence_index < 4:
                    st.session_state.current_sentence_index += 1
                    st.rerun()
                else:
                    st.session_state.current_sentence_index = 0
                    st.session_state.current_crop_index += 1
                    st.rerun()
        
        # Show current results
        if st.session_state.test_results:
            st.subheader("📊 Current Results")
            results_df = pd.DataFrame(st.session_state.test_results)
            st.dataframe(results_df, use_container_width=True)
    
    else:
        # Testing Complete
        st.header("🎉 Testing Complete!")
        
        # Results summary
        total_tests = len(st.session_state.test_results)
        correct_tests = sum(1 for result in st.session_state.test_results if result['is_correct'])
        accuracy = (correct_tests / total_tests * 100) if total_tests > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", total_tests)
        with col2:
            st.metric("Correct", correct_tests)
        with col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        # Download results
        if st.session_state.test_results:
            st.subheader("📥 Download Results")
            
            # Create CSV for download
            results_df = pd.DataFrame(st.session_state.test_results)
            
            # Add QA name and format for download
            results_df['QA_NAME'] = st.session_state.qa_name
            results_df['LANGUAGE'] = st.session_state.selected_language
            results_df['PROJECT'] = st.session_state.selected_project
            
            # Reorder columns
            download_df = results_df[['QA_NAME', 'LANGUAGE', 'PROJECT', 'crop_name', 'crop_code', 'sentence', 'extracted_keyword', 'is_correct', 'timestamp']]
            
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV Results",
                data=csv,
                file_name=f"asr_test_results_{st.session_state.qa_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Reset button
        if st.button("🔄 Start New Test", type="primary"):
            st.session_state.qa_name = ""
            st.session_state.selected_language = ""
            st.session_state.selected_project = ""
            st.session_state.crop_data = []
            st.session_state.current_crops = []
            st.session_state.selected_crops = []
            st.session_state.current_crop_index = 0
            st.session_state.current_sentence_index = 0
            st.session_state.test_results = []
            st.session_state.testing_active = False
            st.rerun()

# Mock functions (to be replaced with actual implementations)
def mock_keyword_extraction(sentence, expected_keyword):
    """Mock keyword extraction - replace with actual NLP implementation"""
    # Simple mock - in real implementation, use actual ASR + NLP
    import random
    mock_results = [expected_keyword, expected_keyword.lower(), expected_keyword.upper(), "Not found"]
    return random.choice(mock_results)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎤 ASR Testing Platform - DCS | Built with Streamlit</p>
    <p>For testing crop name recognition accuracy</p>
</div>
""", unsafe_allow_html=True)

