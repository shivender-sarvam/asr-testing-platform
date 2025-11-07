import streamlit as st
import pandas as pd
import json
import base64
from datetime import datetime
import io
import re
import requests
import os
from urllib.parse import urlencode, parse_qs
import streamlit.components.v1 as components

# Import Azure service for saving results
# Catch ALL exceptions (not just ImportError) because azure_service might raise KeyError
# if Streamlit secrets aren't available during import
try:
    from azure_service import upload_single_test_result, upload_asr_test_results
    AZURE_AVAILABLE = True
except Exception as e:
    AZURE_AVAILABLE = False
    # Don't show warning during import - it might not be initialized yet
    # Warning will be shown later if needed

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
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'session_id' not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if 'current_attempt' not in st.session_state:
    st.session_state.current_attempt = {}  # Track attempts per crop: {crop_index: attempt_number}

# Google OAuth Configuration - will be read in functions
def get_google_config():
    """Get Google OAuth configuration from secrets"""
    try:
        # Try to access secrets directly
        try:
            client_id = st.secrets['GOOGLE_ID']
            client_secret = st.secrets['GOOGLE_SECRET'] 
            redirect_uri = st.secrets['GOOGLE_REDIRECT_URI']
        except:
            # Try nested access
            try:
                client_id = st.secrets.secrets['GOOGLE_ID']
                client_secret = st.secrets.secrets['GOOGLE_SECRET']
                redirect_uri = st.secrets.secrets['GOOGLE_REDIRECT_URI']
            except:
                # Use environment variables as fallback
                client_id = os.environ.get('GOOGLE_ID', 'your-google-client-id')
                client_secret = os.environ.get('GOOGLE_SECRET', 'your-google-client-secret')
                # Use localhost for local testing, Streamlit Cloud URL for production
                if 'localhost' in os.environ.get('STREAMLIT_SERVER_URL', '') or '127.0.0.1' in os.environ.get('STREAMLIT_SERVER_URL', ''):
                    redirect_uri = 'http://localhost:8501/'
                else:
                    redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', st.secrets.get('GOOGLE_REDIRECT_URI', 'https://asr-testing-platform.streamlit.app/'))
        
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri
        }
    except Exception as e:
        return {
            'client_id': 'your-google-client-id',
            'client_secret': 'your-google-client-secret', 
            'redirect_uri': redirect_uri
        }

# Allowed email domains
ALLOWED_DOMAINS = ['gmail.com', 'googlemail.com', 'google.com', 'sarvam.ai']

# Saaras API Configuration (matching Flask/Render version)
SARVAM_API_URL = "http://103.207.148.23/saaras_v2_6/audio/transcriptions"
MODEL_NAME = "/models/saaras-raft-wp20-base-v2v-v2-chunk_5-main-bs64/1-gpu"
BCP47_CODES = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "pa": "pa-IN"
}

def call_sarvam_asr(audio_bytes, language_code, api_key=None, audio_format='wav'):
    """
    Call Sarvam ASR API to transcribe audio
    
    Args:
        audio_bytes: Audio file bytes
        language_code: Language code (e.g., 'hi', 'en')
        api_key: Sarvam API key (from secrets or env)
    
    Returns:
        str: Transcribed text or None if error
    """
    try:
        # Get API key from secrets or environment
        if not api_key:
            try:
                api_key = st.secrets.get('SARVAM_API_KEY', '')
            except:
                try:
                    api_key = st.secrets.secrets.get('SARVAM_API_KEY', '')
                except:
                    api_key = os.environ.get('SARVAM_API_KEY', '')
        
        if not api_key:
            st.error("❌ SARVAM_API_KEY not found. Check Streamlit Cloud secrets.")
            return None
        
        # Get BCP47 language code
        bcp47_lang = BCP47_CODES.get(language_code.lower(), "hi-IN")
        
        # Prepare request - use correct format
        mime_type_for_api = 'audio/wav' if audio_format == 'wav' else 'audio/webm'
        filename = f'audio.{audio_format}'
        
        files = {
            'file': (filename, audio_bytes, mime_type_for_api)
        }
        
        data = {
            'model': MODEL_NAME,
            'language_code': bcp47_lang
        }
        
        headers = {
            'api-subscription-key': api_key
        }
        
        # Debug logging
        st.info(f"🔍 Calling ASR API: {SARVAM_API_URL}, Model: {MODEL_NAME}, Language: {bcp47_lang}, Format: {audio_format}")
        
        # Make API call
        response = requests.post(SARVAM_API_URL, files=files, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                result = response.json()
                transcript = result.get('transcript', result.get('text', '')).strip()
                if transcript:
                    st.success(f"✅ ASR Success: '{transcript}'")
                return transcript
            except Exception as json_error:
                st.error(f"❌ Failed to parse API response: {json_error}")
                st.code(f"Response: {response.text[:500]}")
                return None
        else:
            st.error(f"❌ API error: {response.status_code}")
            try:
                error_body = response.json()
                st.error(f"Error details: {error_body}")
            except:
                st.error(f"Error text: {response.text[:500]}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("API request timed out")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API")
        return None
    except Exception as e:
        st.error(f"ASR API error: {str(e)}")
        return None

def check_match(expected_text, actual_text):
    """
    Check if ASR transcription matches the expected crop name
    EXACT FLASK VERSION
    """
    if not actual_text:
        return False
    
    transcript_lower = actual_text.lower().strip()
    crop_lower = expected_text.lower().strip()
    
    # Direct match
    if crop_lower in transcript_lower:
        return True
    
    # Check for partial matches (useful for compound words)
    crop_words = crop_lower.split()
    transcript_words = transcript_lower.split()
    
    # If crop name has multiple words, check if all words are present
    if len(crop_words) > 1:
        return all(word in transcript_lower for word in crop_words)
    
    # For single words, check if it's a substring
    return crop_lower in transcript_lower

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
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Platform Features:**")
    st.markdown("""
    - Test ASR accuracy with crop names
    - Record audio for multiple languages
    - Generate detailed CSV reports
    - Track pronunciation accuracy
    """)
    
    # Google OAuth Login
    config = get_google_config()
    
    # Debug: Show what we're using (first few chars only for security)
    with st.expander("🔍 Debug OAuth Config", expanded=False):
        st.write(f"Client ID: {config['client_id'][:20]}..." if len(config['client_id']) > 20 else f"Client ID: {config['client_id']}")
        st.write(f"Redirect URI: {config['redirect_uri']}")
        st.write(f"Client ID length: {len(config['client_id'])}")
    
    if config['client_id'] != 'your-google-client-id' and config['client_id']:
        # URL encode the redirect URI properly
        from urllib.parse import quote
        redirect_uri_encoded = quote(config['redirect_uri'], safe='')
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?client_id={config['client_id']}&redirect_uri={redirect_uri_encoded}&scope=openid%20email%20profile&response_type=code"
        
        st.markdown(f"""
        <div style="text-align: center;">
            <a href="{auth_url}" class="google-button">Continue with Google</a>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Google OAuth not configured. Please set GOOGLE_ID and GOOGLE_SECRET in Streamlit secrets.")
        st.info("""
        **To fix this:**
        1. Go to Streamlit Cloud → Settings → Secrets
        2. Make sure you have:
           - `GOOGLE_ID` = your Google OAuth Client ID
           - `GOOGLE_SECRET` = your Google OAuth Client Secret
           - `GOOGLE_REDIRECT_URI` = https://asr-testing-platform.streamlit.app/
        3. Make sure the redirect URI matches exactly in Google Cloud Console
        """)
    
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem;">
        <p>By signing in, you agree to our privacy policy.</p>
        <p><a href="#" style="color: #666;">Privacy</a> - <a href="#" style="color: #666;">Terms</a></p>
    </div>
    """, unsafe_allow_html=True)

def handle_oauth_callback():
    """Handle OAuth callback"""
    query_params = st.query_params
    
    # If already authenticated, don't process callback again
    if st.session_state.authenticated:
        # Clear OAuth code from URL if present
        if 'code' in query_params:
            st.query_params.clear()
        return
    
    # Only process callback if we have a code and are not already authenticated
    if 'code' in query_params and not st.session_state.authenticated:
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
                    # Auto-set QA name from Google OAuth (use name or extract from email)
                    if user_info.get('name'):
                        st.session_state.qa_name = user_info['name']
                    else:
                        # Extract name from email (e.g., "shivender" from "shivender@sarvam.ai")
                        st.session_state.qa_name = user_info['email'].split('@')[0].title()
                    # Clear the OAuth code from URL
                    st.query_params.clear()
                    st.rerun()
                else:
                    st.error("Access denied. Please use a Google account or Sarvam email.")
            else:
                # Don't show error if code was already used (user is already logged in)
                error_msg = token_json.get('error_description', 'Authentication failed')
                if 'invalid_grant' not in error_msg.lower():
                    st.error(f"Authentication failed: {error_msg}")
        except Exception as e:
            # Only show error if not already authenticated
            if not st.session_state.authenticated:
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
    # Auto-set QA name from user_info if not set
    if st.session_state.user_info and not st.session_state.qa_name:
        if st.session_state.user_info.get('name'):
            st.session_state.qa_name = st.session_state.user_info['name']
        else:
            # Extract name from email (e.g., "shivender" from "shivender@sarvam.ai")
            st.session_state.qa_name = st.session_state.user_info['email'].split('@')[0].title()
    
    # DEBUG: Show current app state
    with st.sidebar.expander("🔍 DEBUG: App State", expanded=False):
        st.write(f"**Has language:** {bool(st.session_state.selected_language)}")
        st.write(f"**Has test_data:** {bool(st.session_state.test_data)}")
        st.write(f"**Test data length:** {len(st.session_state.test_data) if st.session_state.test_data else 0}")
        st.write(f"**Current index:** {st.session_state.current_test_index}")
    
    if not st.session_state.selected_language:
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
    """Step 1: Language Selection"""
    st.header("🌍 Step 1: Select Language")
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
    """Step 2: CSV Upload"""
    st.header("📁 Step 2: Upload CSV or Start Testing")
    
    st.markdown("Upload a CSV file with crop data in the format: `serial_number,crop_code,crop_name,language,project`")
    st.info("💡 **Tip:** Column names can be flexible (e.g., 'crop name', 'Crop Name', 'crop_name' all work)")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Normalize column names (handle variations)
            df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
            
            # Filter by selected language if language column exists
            if 'language' in df.columns:
                df = df[df['language'].str.lower() == st.session_state.selected_language.lower()]
                if df.empty:
                    st.warning(f"No crops found for language: {st.session_state.selected_language}")
                else:
                    st.success(f"Found {len(df)} crops for {st.session_state.selected_language}!")
            else:
                st.success(f"CSV loaded successfully! Found {len(df)} crops.")
            
            st.dataframe(df.head())
            
            if st.button("Start Testing", type="primary"):
                if not df.empty:
                    st.session_state.test_data = df.to_dict('records')
                    st.rerun()
                else:
                    st.error("No data to test. Please check your CSV file.")
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
    """Main testing interface - matches Flask/Render version exactly"""
    # DEBUG: Check if we're in testing interface - ALWAYS VISIBLE
    st.error("🔍🔍🔍 DEBUG MODE ACTIVE - Testing Interface Loaded 🔍🔍🔍")
    st.info(f"Current index: {st.session_state.current_test_index}, Total crops: {len(st.session_state.test_data)}")
    
    if st.session_state.current_test_index < len(st.session_state.test_data):
        current_crop = st.session_state.test_data[st.session_state.current_test_index]
        
        # Handle different column name formats
        crop_name = current_crop.get('crop_name') or current_crop.get('name', 'Unknown')
        crop_code = current_crop.get('crop_code') or current_crop.get('code', 'N/A')
        language = current_crop.get('language', st.session_state.selected_language or 'en')
        
        st.header(f"🎤 ASR Testing Session")
        
        # Progress info
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**QA Name:** {st.session_state.qa_name}")
            st.markdown(f"**Language:** {language.title() if isinstance(language, str) else language}")
        with col2:
            st.markdown(f"**Progress:** {st.session_state.current_test_index + 1} / {len(st.session_state.test_data)}")
            st.markdown(f"**Current Crop:** {crop_name}")
        
        # Progress bar
        progress = (st.session_state.current_test_index + 1) / len(st.session_state.test_data)
        st.progress(progress)
        
        # Crop name display
        st.info(f"🌾 **Crop Name:** {crop_name}")
        st.markdown(f"*You need to record 5 audio samples using this crop name in sentences.*")
        
        # Initialize attempt tracking for this crop
        crop_index = st.session_state.current_test_index
        
        # Check if user wants to increment attempt (from "Record Again" button)
        if st.query_params.get('increment_attempt') == 'true':
            current_attempt_num = st.session_state.current_attempt.get(crop_index, 1)
            if current_attempt_num < 5:
                st.session_state.current_attempt[crop_index] = current_attempt_num + 1
                # Clear previous attempt's state
                old_recording_key = f"recording_{crop_index}_attempt_{current_attempt_num}"
                old_audio_submitted_key = f"audio_submitted_{old_recording_key}"
                keys_to_clear = [f'audio_upload_{old_recording_key}', f'audio_processed_{old_recording_key}', 
                               f'asr_result_{old_recording_key}', old_audio_submitted_key]
                for key in keys_to_clear:
                    if key in st.session_state:
                        del st.session_state[key]
            st.query_params.pop('increment_attempt', None)
            st.rerun()
        
        current_attempt_num = st.session_state.current_attempt.get(crop_index, 1)
        max_attempts = 5
        
        # Show attempt counter (like Flask)
        st.markdown(f"**Recording Attempt: {current_attempt_num} / {max_attempts}**")
        
        # Initialize session state for this recording
        recording_key = f"recording_{crop_index}_attempt_{current_attempt_num}"
        audio_submitted_key = f"audio_submitted_{recording_key}"
        
        # Audio Recording Component
        st.markdown("### 🎤 Record Audio")
        
        # Custom Audio Recorder HTML Component
        # Use .format() instead of f-string to avoid issues with curly braces
        audio_recorder_html = """
        <div id="audio-recorder-{recording_key}" style="text-align: center; padding: 20px;">
            <button id="startBtn-{recording_key}" style="
                background: #1e3c72;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            ">🎤 Start Recording</button>
            
            <button id="stopBtn-{recording_key}" disabled style="
                background: #dc3545;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 16px;
                border-radius: 5px;
                cursor: not-allowed;
                margin: 5px;
                opacity: 0.5;
            ">⏹️ Stop Recording</button>
            
            <div id="status-{recording_key}" style="margin: 10px 0; font-weight: bold; color: #dc3545; display: none;">
                🔴 Recording... Speak clearly!
            </div>
            
            <div id="playback-{recording_key}" style="margin: 20px 0; display: none;">
                <p style="color: #28a745; font-weight: bold;">✅ Recording completed! Listen to your recording:</p>
                <audio id="audioPlayer-{recording_key}" controls style="width: 100%; max-width: 500px; margin: 10px auto; display: block;"></audio>
                <div style="margin: 15px 0;">
                    <button id="submitBtn-{recording_key}" style="
                        background: #007bff;
                        color: white;
                        border: none;
                        padding: 12px 25px;
                        font-size: 16px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 5px;
                    ">📤 Submit Recording</button>
                    <button id="recordAgainBtn-{recording_key}" style="
                        background: #6c757d;
                        color: white;
                        border: none;
                        padding: 12px 25px;
                        font-size: 16px;
                        border-radius: 5px;
                        cursor: pointer;
                        margin: 5px;
                    ">🔄 Record Again</button>
                </div>
            </div>
        </div>
        
        <script>
        (function() {{
            const key = '{recording_key}';
            let mediaRecorder;
            let audioChunks = [];
            let audioBlob = null;
            
            const startBtn = document.getElementById('startBtn-' + key);
            const stopBtn = document.getElementById('stopBtn-' + key);
            const statusDiv = document.getElementById('status-' + key);
            const playbackDiv = document.getElementById('playback-' + key);
            const audioPlayer = document.getElementById('audioPlayer-' + key);
            const submitBtn = document.getElementById('submitBtn-' + key);
            const recordAgainBtn = document.getElementById('recordAgainBtn-' + key);
            
            startBtn.addEventListener('click', async function() {{
                try {{
                    const stream = await navigator.mediaDevices.getUserMedia({{
                        audio: {{
                            sampleRate: 16000,
                            channelCount: 1,
                            echoCancellation: true,
                            noiseSuppression: true
                        }}
                    }});
                    
                    let mimeType = 'audio/webm';
                    if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {{
                        mimeType = 'audio/webm;codecs=opus';
                    }}
                    
                    mediaRecorder = new MediaRecorder(stream, {{ mimeType: mimeType }});
                    audioChunks = [];
                    
                    mediaRecorder.ondataavailable = function(event) {{
                        if (event.data.size > 0) {{
                            audioChunks.push(event.data);
                        }}
                    }};
                    
                    mediaRecorder.onstop = async function() {{
                        audioBlob = new Blob(audioChunks, {{ type: mimeType }});
                        const audioUrl = URL.createObjectURL(audioBlob);
                        audioPlayer.src = audioUrl;
                        playbackDiv.style.display = 'block';
                        
                        // Store audio blob globally for submit button
                        window['audioBlob_' + key] = audioBlob;
                        console.log('Audio blob stored, ready for submission');
                        
                        stream.getTracks().forEach(track => track.stop());
                    }};
                    
                    // Submit button handler - DOWNLOAD FILE (MOST RELIABLE)
                    submitBtn.addEventListener('click', async function() {{
                        if (!audioBlob) {{
                            alert('No recording to submit');
                            return;
                        }}
                        
                        submitBtn.disabled = true;
                        submitBtn.textContent = '⏳ Downloading...';
                        
                        // Download audio file - user will upload it
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const a = document.createElement('a');
                        a.href = audioUrl;
                        a.download = 'recording_' + key + '.webm';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(audioUrl);
                        
                        submitBtn.textContent = '✅ Downloaded! Upload below';
                        submitBtn.style.background = '#28a745';
                    }});
                    
                    // Record again button - always increment attempt if results are shown
                    recordAgainBtn.addEventListener('click', function() {{
                        // Check if results section exists (means result was submitted)
                        const resultsSection = document.querySelector('[data-testid*="stMarkdownContainer"]');
                        const hasResults = resultsSection && resultsSection.textContent.includes('Results');
                        
                        // Always try to increment attempt (safer approach)
                        const currentUrl = window.location.href;
                        const url = new URL(currentUrl);
                        url.searchParams.set('increment_attempt', 'true');
                        url.searchParams.set('_t', Date.now()); // Force reload
                        
                        // Update URL to trigger Streamlit rerun with incremented attempt
                        if (window.parent && window.parent !== window) {{
                            try {{
                                window.parent.location.href = url.toString();
                            }} catch(e) {{
                                // Cross-origin - use postMessage
                                window.parent.postMessage({{
                                    type: 'streamlit:increment_attempt',
                                    key: key
                                }}, '*');
                                window.location.href = url.toString();
                            }}
                        }} else {{
                            window.location.href = url.toString();
                        }}
                    }});
                    
                    mediaRecorder.start(100);
                    startBtn.disabled = true;
                    startBtn.style.opacity = '0.5';
                    startBtn.style.cursor = 'not-allowed';
                    stopBtn.disabled = false;
                    stopBtn.style.opacity = '1';
                    stopBtn.style.cursor = 'pointer';
                    statusDiv.style.display = 'block';
                    
                }} catch (error) {{
                    alert('Error accessing microphone: ' + error.message);
                }}
            }});
            
            stopBtn.addEventListener('click', function() {{
                if (mediaRecorder && mediaRecorder.state === 'recording') {{
                    mediaRecorder.stop();
                    startBtn.disabled = false;
                    startBtn.style.opacity = '1';
                    startBtn.style.cursor = 'pointer';
                    stopBtn.disabled = true;
                    stopBtn.style.opacity = '0.5';
                    stopBtn.style.cursor = 'not-allowed';
                    statusDiv.style.display = 'none';
                }}
            }});
        }})();
        </script>
        """.format(recording_key=recording_key)
        
        # Render the audio recorder
        components.html(audio_recorder_html, height=300)
        
        # File uploader for audio (SIMPLE & RELIABLE)
        st.markdown("---")
        st.markdown("**📤 Upload your recorded audio:**")
        uploaded_audio = st.file_uploader(
            "Choose audio file",
            type=['webm', 'wav', 'mp3'],
            key=f"audio_upload_{recording_key}",
            help="After recording, click Submit to download, then upload the file here"
        )
        
        # Process uploaded audio
        audio_bytes = None
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            st.success("✅ Audio file received!")
        
        # Process audio when uploaded (EXACT FLASK WORKFLOW)
        if audio_bytes and not st.session_state.get(f'audio_processed_{recording_key}', False):
            with st.spinner("🔄 Processing audio with ASR..."):
                try:
                    # Determine audio format
                    audio_format = 'wav'
                    if uploaded_audio.name.endswith('.webm'):
                        try:
                            from pydub import AudioSegment
                            import io
                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                            wav_buffer = io.BytesIO()
                            audio_segment.export(wav_buffer, format="wav")
                            audio_bytes = wav_buffer.getvalue()
                            audio_format = 'wav'
                        except Exception as e:
                            st.warning(f"Could not convert webm to wav: {e}. Using original format.")
                            audio_format = 'webm'
                    
                    # Call ASR API (like Flask)
                    api_key = st.secrets.get('SARVAM_API_KEY', os.environ.get('SARVAM_API_KEY', None))
                    if not api_key:
                        st.error("❌ SARVAM_API_KEY not found in secrets. Please configure it in Streamlit Cloud secrets.")
                        st.session_state[f'asr_result_{recording_key}'] = {
                            'transcript': None,
                            'matches': False,
                            'error': 'API key not configured'
                        }
                    else:
                        asr_transcript = call_sarvam_asr(
                            audio_bytes,
                            language,
                            api_key,
                            audio_format=audio_format
                        )
                        
                        if asr_transcript:
                            # Use Flask's exact keyword matching logic
                            matches = check_match(crop_name, asr_transcript)
                            st.session_state[f'asr_result_{recording_key}'] = {
                                'transcript': asr_transcript,
                                'matches': matches
                            }
                        else:
                            # ASR failed - store error
                            st.session_state[f'asr_result_{recording_key}'] = {
                                'transcript': None,
                                'matches': False,
                                'error': 'ASR API returned no transcript'
                            }
                    
                    # Mark as processed (even if failed, so we don't retry infinitely)
                    st.session_state[f'audio_processed_{recording_key}'] = True
                    st.session_state[audio_submitted_key] = True
                    st.rerun()
                except Exception as e:
                    import traceback
                    error_traceback = traceback.format_exc()
                    st.error(f"❌ Error processing audio: {e}")
                    # Store full error details
                    st.session_state[f'asr_result_{recording_key}'] = {
                        'transcript': None,
                        'matches': False,
                        'error': str(e),
                        'traceback': error_traceback
                    }
                    st.session_state[f'audio_processed_{recording_key}'] = True
                    st.session_state[audio_submitted_key] = True
        
        # DEBUG: Show what's happening
        with st.expander("🔍 Debug Info", expanded=False):
            st.write(f"**Audio uploaded:** {uploaded_audio is not None}")
            st.write(f"**Audio processed:** {st.session_state.get(f'audio_processed_{recording_key}', False)}")
            st.write(f"**Audio submitted:** {st.session_state.get(audio_submitted_key, False)}")
            st.write(f"**Has result:** {bool(st.session_state.get(f'asr_result_{recording_key}', {}))}")
            st.write(f"**Test results count:** {len(st.session_state.test_results)}")
            st.write(f"**Current attempt:** {current_attempt_num} / {max_attempts}")
        
        # Show results immediately after processing (EXACT FLASK WORKFLOW)
        if st.session_state.get(audio_submitted_key, False):
            result = st.session_state.get(f'asr_result_{recording_key}', {})
            st.markdown("---")
            st.markdown("### ✅ Results")
            
            if result.get('transcript'):
                st.markdown(f"**Transcription:** {result['transcript']}")
                keyword_badge = "✅ **Yes**" if result.get('matches') else "❌ **No**"
                st.markdown(f"**Keyword Detected:** {keyword_badge}")
                
                # Save result (per attempt, like Flask)
                result_saved_key = f'result_saved_{recording_key}'
                if not st.session_state.get(result_saved_key, False):
                    test_result = {
                        "qa_name": st.session_state.qa_name,
                        "qa_email": st.session_state.user_info.get('email', '') if st.session_state.user_info else '',
                        "session_id": st.session_state.session_id,
                        "crop_name": crop_name,
                        "crop_code": crop_code,
                        "language": language,
                        "attempt_number": current_attempt_num,
                        "expected": crop_name,
                        "transcript": result['transcript'],
                        "keyword_detected": "Yes" if result.get('matches') else "No",
                        "match": "Yes" if result.get('matches') else "No",  # Keep for compatibility
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "audio_recorded": True
                    }
                    # Save to session state
                    st.session_state.test_results.append(test_result)
                    st.session_state[result_saved_key] = True
                    
                    # IMMEDIATELY save to Azure (like Flask) - prevents data loss
                    if AZURE_AVAILABLE:
                        try:
                            user_email = st.session_state.user_info.get('email', 'unknown@example.com') if st.session_state.user_info else 'unknown@example.com'
                            azure_url = upload_single_test_result(
                                test_result=test_result,
                                user_email=user_email,
                                language=language,
                                session_id=st.session_state.session_id
                            )
                            st.success(f"✅ Saved to Azure! Total: {len(st.session_state.test_results)}")
                        except Exception as e:
                            st.warning(f"⚠️ Saved locally but Azure failed: {e}")
                            st.success(f"✅ Result saved locally! Total: {len(st.session_state.test_results)}")
                    else:
                        st.success(f"✅ Result saved! Total: {len(st.session_state.test_results)}")
            
            # AUTO-ADVANCE like Flask (after showing results)
            st.markdown("---")
            if result.get('transcript'):
                if current_attempt_num < max_attempts:
                    # More attempts remaining - show buttons to continue
                    st.info(f"✅ Result saved! Continue with attempt {current_attempt_num + 1}/{max_attempts}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"➡️ Continue to Attempt {current_attempt_num + 1}", type="primary", key=f"continue_attempt_{recording_key}", use_container_width=True):
                            # Move to next attempt
                            st.session_state.current_attempt[crop_index] = current_attempt_num + 1
                            # Clear this attempt's state
                            result_saved_key = f'result_saved_{recording_key}'
                            keys_to_clear = [f'audio_upload_{recording_key}', f'audio_processed_{recording_key}', 
                                           f'asr_result_{recording_key}', audio_submitted_key, result_saved_key]
                            for key in keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                    with col2:
                        if st.button("🔄 Record Again", type="secondary", key=f"record_again_streamlit_{recording_key}", use_container_width=True):
                            # Increment attempt and clear state
                            if current_attempt_num < max_attempts:
                                st.session_state.current_attempt[crop_index] = current_attempt_num + 1
                            result_saved_key = f'result_saved_{recording_key}'
                            # Clear all state for this attempt
                            keys_to_clear = [f'audio_upload_{recording_key}', f'audio_processed_{recording_key}', 
                                           f'asr_result_{recording_key}', audio_submitted_key, result_saved_key]
                            for key in keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                            st.rerun()
                else:
                    # All attempts done - show button to move to next crop
                    st.info("✅ All 5 attempts completed for this crop!")
                    if crop_index + 1 < len(st.session_state.test_data):
                        if st.button("➡️ Next Crop", type="primary", key=f"next_crop_{recording_key}", use_container_width=True):
                            # Move to next crop
                            st.session_state.current_test_index = crop_index + 1
                            # Reset attempt for new crop
                            if crop_index + 1 not in st.session_state.current_attempt:
                                st.session_state.current_attempt[crop_index + 1] = 1
                            # Clear all state
                            for key in list(st.session_state.keys()):
                                if 'recording_' in key or 'audio_' in key:
                                    del st.session_state[key]
                            st.rerun()
                    else:
                        # All crops done - go to results
                        st.success("🎉 All crops completed!")
                        if st.button("📊 View Results & Download CSV", type="primary", key=f"view_results_{recording_key}", use_container_width=True):
                            st.session_state.show_results = True
                            st.rerun()
            else:
                # Show specific error if available
                error_msg = result.get('error', 'Unknown error')
                st.error(f"❌ ASR processing failed: {error_msg}")
                
                # Show detailed error logs
                with st.expander("📋 **Error Logs**", expanded=True):
                    st.write("**Error Details:**")
                    st.code(f"Error: {error_msg}", language='text')
                    
                    # Show full traceback if available
                    if result.get('traceback'):
                        st.write("**Full Traceback:**")
                        st.code(result.get('traceback'), language='python')
                    
                    # Show API key status
                    st.write("**API Key Status:**")
                    try:
                        api_key_check = st.secrets.get('SARVAM_API_KEY', 'NOT FOUND')
                        if api_key_check == 'NOT FOUND':
                            try:
                                api_key_check = st.secrets.secrets.get('SARVAM_API_KEY', 'NOT FOUND')
                            except:
                                api_key_check = os.environ.get('SARVAM_API_KEY', 'NOT FOUND')
                        
                        if api_key_check != 'NOT FOUND':
                            st.success(f"✅ API Key found: {api_key_check[:10]}...{api_key_check[-5:]}")
                        else:
                            st.error("❌ API Key NOT FOUND in secrets or environment")
                            st.info("💡 Add SARVAM_API_KEY to Streamlit Cloud secrets")
                    except Exception as e:
                        st.error(f"❌ Error checking API key: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language='python')
                    
                    # Show API configuration
                    st.write("**API Configuration:**")
                    st.code(f"URL: {SARVAM_API_URL}\nModel: {MODEL_NAME}\nLanguage: {language}\nBCP47 Code: {BCP47_CODES.get(language.lower(), 'hi-IN')}", language='text')
                    
                    # Show audio info
                    if uploaded_audio:
                        st.write("**Audio File Info:**")
                        audio_size = len(audio_bytes) if 'audio_bytes' in locals() else 'N/A'
                        st.code(f"Name: {uploaded_audio.name}\nSize: {audio_size} bytes\nType: {uploaded_audio.type}", language='text')
                
                if st.button("🔄 Try Again", key=f"retry_{recording_key}"):
                    # Clear state to allow retry
                    keys_to_clear = [f'audio_upload_{recording_key}', f'audio_processed_{recording_key}', 
                                   f'asr_result_{recording_key}', audio_submitted_key]
                    for key in keys_to_clear:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        # End Session button (matches Flask version)
        st.markdown("---")
        if st.button("🏁 End Session & Get Reports", type="primary"):
            st.session_state.show_results = True
            st.rerun()
    
    # Show results page
    if st.session_state.get('show_results', False) or (st.session_state.test_data and st.session_state.current_test_index >= len(st.session_state.test_data)):
        # Testing complete
        st.header("🎉 Testing Complete!")
        
        # Results summary
        if st.session_state.test_results:
            st.markdown(f"**Total Tests:** {len(st.session_state.test_results)}")
            # Count matches (handle both old 'accuracy' format and new 'match' format)
            matches_count = sum(1 for r in st.session_state.test_results if r.get('match') == 'Yes' or (r.get('match') is True) or (r.get('keyword_detected') == 'Yes') or (r.get('accuracy', 0) > 0))
            st.markdown(f"**Matches:** {matches_count} / {len(st.session_state.test_results)}")
        else:
            st.warning("⚠️ No test results found. Please complete at least one test.")
        
        # Results table - format like Flask
        if st.session_state.test_results:
            # Create DataFrame with Flask-compatible columns
            results_list = []
            for r in st.session_state.test_results:
                results_list.append({
                    'QA Name': r.get('qa_name', ''),
                    'Language': r.get('language', ''),
                    'Session ID': r.get('session_id', st.session_state.session_id),
                    'Crop Name': r.get('crop_name', ''),
                    'Attempt Number': r.get('attempt_number', 1),
                    'Transcription': r.get('transcript', r.get('actual', '')),
                    'Keyword Detected': r.get('keyword_detected', r.get('match', 'No')),
                    'Timestamp': r.get('timestamp', '')
                })
            
            results_df = pd.DataFrame(results_list)
            st.markdown("### 📊 Test Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download CSV - Flask format
            st.markdown("---")
            csv = results_df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            email = st.session_state.user_info.get('email', 'unknown').replace('@', '_at_') if st.session_state.user_info else 'unknown'
            language = st.session_state.selected_language or 'unknown'
            session_id = st.session_state.session_id
            filename = f"asr_test_results_{email}_{language}_{session_id}_{timestamp}.csv"
            
            # Auto-upload to Azure when results page is shown (like Flask)
            azure_upload_key = f'azure_uploaded_{session_id}'
            if AZURE_AVAILABLE and not st.session_state.get(azure_upload_key, False):
                try:
                    user_email = st.session_state.user_info.get('email', 'unknown@example.com') if st.session_state.user_info else 'unknown@example.com'
                    azure_url = upload_asr_test_results(
                        test_results=st.session_state.test_results,
                        user_email=user_email,
                        language=language,
                        session_id=session_id
                    )
                    st.session_state[azure_upload_key] = True
                    st.success(f"✅ All results saved to Azure automatically!")
                except Exception as e:
                    st.warning(f"⚠️ Azure upload failed: {e}. Results saved locally.")
            
            # Download CSV
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                type="primary",
                use_container_width=True
            )
        else:
            st.warning("⚠️ No test results found. Please complete at least one test.")
        
        # Reset option
        if st.button("🔄 Start New Test", type="primary"):
            st.session_state.qa_name = None
            st.session_state.selected_language = None
            st.session_state.test_data = []
            st.session_state.current_test_index = 0
            st.session_state.test_results = []
            st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.session_state.current_attempt = {}
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
