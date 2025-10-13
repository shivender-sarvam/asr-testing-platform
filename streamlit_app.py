import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
import io
import re
import requests
import os
from urllib.parse import urlencode

# Page config
st.set_page_config(
    page_title="ASR Testing Platform - Sarvam",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Sarvam branding
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sarvam-brand {
        font-size: 2rem;
        font-weight: bold;
        color: #1e3c72;
        margin-bottom: 1rem;
    }
    
    .login-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: #fff;
        border: 2px solid #e9ecef;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        border-color: #1e3c72;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.2);
    }
    
    .google-button {
        background: #4285f4;
        color: white;
        padding: 12px 24px;
        border: none;
        border-radius: 5px;
        font-size: 16px;
        cursor: pointer;
        text-decoration: none;
        display: inline-block;
        margin: 10px;
    }
    
    .google-button:hover {
        background: #3367d6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None
if 'qa_name' not in st.session_state:
    st.session_state.qa_name = None
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = []
if 'current_test_index' not in st.session_state:
    st.session_state.current_test_index = 0
if 'test_results' not in st.session_state:
    st.session_state.test_results = []

# Google OAuth Configuration - will be read in functions
def get_google_config():
    """Get Google OAuth configuration from secrets"""
    try:
        # Debug: Show what secrets are available
        st.write("🔍 Debug - Available secrets keys:", list(st.secrets.keys()) if hasattr(st.secrets, 'keys') else "No secrets")
        
        # Try to access secrets directly
        try:
            client_id = st.secrets['GOOGLE_ID']
            client_secret = st.secrets['GOOGLE_SECRET'] 
            redirect_uri = st.secrets['GOOGLE_REDIRECT_URI']
            st.write("🔍 Debug - Direct secrets access worked")
        except:
            # Try nested access
            try:
                client_id = st.secrets.secrets['GOOGLE_ID']
                client_secret = st.secrets.secrets['GOOGLE_SECRET']
                redirect_uri = st.secrets.secrets['GOOGLE_REDIRECT_URI']
                st.write("🔍 Debug - Nested secrets access worked")
            except:
                # Use environment variables as fallback
                client_id = os.environ.get('GOOGLE_ID', 'your-google-client-id')
                client_secret = os.environ.get('GOOGLE_SECRET', 'your-google-client-secret')
                redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'https://your-app.streamlit.app/')
                st.write("🔍 Debug - Using environment variables")
        
        st.write(f"🔍 Debug - Client ID: {client_id[:10]}..." if client_id != 'your-google-client-id' else "🔍 Debug - Client ID: NOT SET")
        
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri
        }
    except Exception as e:
        st.write(f"🔍 Debug - Error reading secrets: {e}")
        return {
            'client_id': 'your-google-client-id',
            'client_secret': 'your-google-client-secret', 
            'redirect_uri': 'https://your-app.streamlit.app/'
        }

# Allowed email domains
ALLOWED_DOMAINS = ['gmail.com', 'googlemail.com', 'google.com', 'sarvam.ai']

def check_authentication():
    """Check if user is authenticated"""
    if not st.session_state.authenticated:
        show_login_page()
        return False
    return True

def show_login_page():
    """Display login page with Sarvam branding"""
    st.markdown("""
    <div class="main-header">
        <h1>🎤 ASR Testing Platform</h1>
        <h2>sarvam</h2>
        <p>QA Testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="login-container">
        <h2>Welcome back</h2>
        <p>Login to your account</p>
        
        <h3>Platform Features:</h3>
        <ul style="text-align: left; display: inline-block;">
            <li>Test ASR accuracy with crop names</li>
            <li>Record audio for multiple languages</li>
            <li>Generate detailed CSV reports</li>
            <li>Track pronunciation accuracy</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Google OAuth Login
    config = get_google_config()
    
    if config['client_id'] != 'your-google-client-id':
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={config['client_id']}&redirect_uri={config['redirect_uri']}&scope=openid%20email%20profile&response_type=code"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="{auth_url}" class="google-button">Continue with Google</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Google OAuth not configured. Please set GOOGLE_ID and GOOGLE_SECRET environment variables.")
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <p>By signing in, you agree to our privacy policy.</p>
        <p><a href="#" style="color: #666;">Privacy</a> - <a href="#" style="color: #666;">Terms</a></p>
    </div>
    """, unsafe_allow_html=True)

def handle_oauth_callback():
    """Handle OAuth callback"""
    query_params = st.query_params
    
    if 'code' in query_params:
        code = query_params['code']
        config = get_google_config()
        
        # Exchange code for token
        token_url = 'https://oauth2.googleapis.com/token'
        token_data = {
            'client_id': config['client_id'],
            'client_secret': config['client_secret'],
            'code': code,
            'grant_type': 'authorization_code',
            'redirect_uri': config['redirect_uri']
        }
        
        try:
            token_response = requests.post(token_url, data=token_data)
            token_json = token_response.json()
            
            if 'access_token' in token_json:
                access_token = token_json['access_token']
                
                # Get user info
                user_info_url = f'https://www.googleapis.com/oauth2/v2/userinfo?access_token={access_token}'
                user_response = requests.get(user_info_url)
                user_info = user_response.json()
                
                # Check domain
                email_domain = user_info['email'].split('@')[1]
                if email_domain in ALLOWED_DOMAINS:
                    st.session_state.authenticated = True
                    st.session_state.user_info = user_info
                    st.success(f"Welcome, {user_info['name']}!")
                    st.rerun()
                else:
                    st.error("Access denied. Please use a Google account or Sarvam email.")
            else:
                st.error("Authentication failed. Please try again.")
        except Exception as e:
            st.error(f"Login failed: {str(e)}")

def main_app():
    """Main application interface"""
    # Header with Sarvam branding
    st.markdown("""
    <div class="main-header">
        <h1>🎤 ASR Testing Platform</h1>
        <h2>sarvam</h2>
        <p>QA Testing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User info
    if st.session_state.user_info:
        st.sidebar.markdown(f"**Welcome, {st.session_state.user_info['name']}**")
        st.sidebar.markdown(f"Email: {st.session_state.user_info['email']}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.rerun()
    
    # Main content
    if not st.session_state.qa_name:
        show_name_input()
    elif not st.session_state.selected_language:
        show_language_selection()
    elif not st.session_state.test_data:
        show_csv_upload()
    else:
        show_testing_interface()

def show_name_input():
    """Step 1: Enter QA Name"""
    st.header("👤 Step 1: Enter Your Name")
    st.markdown("Enter your name to start the ASR testing process.")
    
    qa_name = st.text_input("QA Name", placeholder="Enter your name here...")
    
    if st.button("Continue", type="primary"):
        if qa_name:
            st.session_state.qa_name = qa_name
            st.rerun()
        else:
            st.error("Please enter your name.")

def show_language_selection():
    """Step 2: Language Selection"""
    st.header("🌍 Step 2: Select Language")
    st.markdown("Choose the language for ASR testing.")
    
    languages = {
        "English": "en",
        "Hindi": "hi", 
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Punjabi": "pa"
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_lang = st.selectbox("Select Language", list(languages.keys()))
    
    with col2:
        st.markdown("**Available Languages:**")
        for lang in languages.keys():
            st.write(f"• {lang}")
    
    if st.button("Continue", type="primary"):
        st.session_state.selected_language = languages[selected_lang]
        st.rerun()

def show_csv_upload():
    """Step 3: CSV Upload"""
    st.header("📁 Step 3: Upload CSV or Start Testing")
    
    st.markdown("Upload a CSV file with crop data in the format: `serial_number,crop_code,crop_name,language,project`")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully!")
            st.dataframe(df.head())
            
            if st.button("Start Testing", type="primary"):
                st.session_state.test_data = df.to_dict('records')
                st.rerun()
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
    
    # Sample data option
    st.markdown("---")
    st.markdown("**Or use sample data:**")
    if st.button("Load Sample Data", type="secondary"):
        sample_data = [
            {"serial_number": 1, "crop_code": "RICE001", "crop_name": "Basmati Rice", "language": st.session_state.selected_language, "project": "Sample"},
            {"serial_number": 2, "crop_code": "WHEAT001", "crop_name": "Wheat", "language": st.session_state.selected_language, "project": "Sample"},
            {"serial_number": 3, "crop_code": "CORN001", "crop_name": "Corn", "language": st.session_state.selected_language, "project": "Sample"}
        ]
        st.session_state.test_data = sample_data
        st.rerun()

def show_testing_interface():
    """Main testing interface"""
    if st.session_state.current_test_index < len(st.session_state.test_data):
        current_crop = st.session_state.test_data[st.session_state.current_test_index]
        
        st.header(f"🎤 Testing: {current_crop['name']}")
        st.markdown(f"**Crop Code:** {current_crop['code']} | **Language:** {current_crop['language'].title()}")
        
        # Progress
        progress = (st.session_state.current_test_index + 1) / len(st.session_state.test_data)
        st.progress(progress)
        st.markdown(f"**Progress:** {st.session_state.current_test_index + 1} of {len(st.session_state.test_data)}")
        
        # Test sentence
        test_sentence = f"Please say {current_crop['name']}"
        st.markdown(f"**Say this sentence:** \"{test_sentence}\"")
        
        # Recording interface
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🎤 Start Recording", type="primary"):
                st.info("Recording started... (This is a demo - actual recording would be implemented)")
        
        with col2:
            if st.button("⏹️ Stop Recording"):
                st.success("Recording completed!")
        
        # Navigation
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_test_index > 0:
                st.session_state.current_test_index -= 1
                st.rerun()
        
        with col2:
            if st.button("Skip ➡️"):
                st.session_state.current_test_index += 1
                st.rerun()
        
        with col3:
            if st.button("✅ Complete Test", type="primary"):
                # Simulate test completion
                result = {
                    "crop_name": current_crop['name'],
                    "crop_code": current_crop['code'],
                    "expected": test_sentence,
                    "actual": f"Recorded audio for {current_crop['name']}",
                    "accuracy": 95.0,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.test_results.append(result)
                st.session_state.current_test_index += 1
                st.rerun()
    
    else:
        # Testing complete
        st.header("🎉 Testing Complete!")
        
        # Results summary
        st.markdown(f"**Total Tests:** {len(st.session_state.test_results)}")
        st.markdown(f"**Average Accuracy:** {sum(r['accuracy'] for r in st.session_state.test_results) / len(st.session_state.test_results):.1f}%")
        
        # Results table
        if st.session_state.test_results:
            results_df = pd.DataFrame(st.session_state.test_results)
            st.dataframe(results_df)
            
            # Download CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=f"asr_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Reset option
        if st.button("🔄 Start New Test", type="primary"):
            st.session_state.qa_name = None
            st.session_state.selected_language = None
            st.session_state.test_data = []
            st.session_state.current_test_index = 0
            st.session_state.test_results = []
            st.rerun()

# Main execution
def main():
    # Handle OAuth callback
    handle_oauth_callback()
    
    # Check authentication
    if check_authentication():
        main_app()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🎤 ASR Testing Platform - Sarvam | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
