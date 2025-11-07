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
import streamlit.components.v1 as components

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
                redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'https://your-app.streamlit.app/')
        
        return {
            'client_id': client_id,
            'client_secret': client_secret,
            'redirect_uri': redirect_uri
        }
    except Exception as e:
        return {
            'client_id': 'your-google-client-id',
            'client_secret': 'your-google-client-secret', 
            'redirect_uri': 'https://your-app.streamlit.app/'
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
            st.warning("⚠️ Sarvam API key not configured.")
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
        
        # Make API call
        response = requests.post(SARVAM_API_URL, files=files, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            try:
                result = response.json()
                transcript = result.get('transcript', result.get('text', '')).strip()
                return transcript
            except Exception as json_error:
                st.error(f"Failed to parse API response: {json_error}")
                return None
        else:
            st.error(f"API error: {response.status_code}")
            try:
                error_body = response.json()
                st.error(f"Error: {error_body}")
            except:
                st.error(f"Error: {response.text[:200]}")
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
    
    Args:
        expected_text: Expected crop name (e.g., "Wheat")
        actual_text: ASR transcription result
    
    Returns:
        bool: True if match, False if no match
    """
    if not actual_text:
        return False
    
    # Normalize both texts (lowercase, remove extra spaces and punctuation)
    expected_clean = re.sub(r'[^\w\s]', '', expected_text.lower().strip())
    actual_clean = re.sub(r'[^\w\s]', '', actual_text.lower().strip())
    
    # Exact match
    if expected_clean == actual_clean:
        return True
    
    # Check if expected word is in the transcription
    if expected_clean in actual_clean:
        return True
    
    # Check if transcription contains key parts of expected word
    expected_words = expected_clean.split()
    for word in expected_words:
        if word in actual_clean:
            return True
    
    # No match
    return False

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
                    
                    // Submit button handler - matches Flask version
                    submitBtn.addEventListener('click', async function() {{
                        if (!audioBlob) {{
                            alert('No recording to submit');
                            return;
                        }}
                        
                        submitBtn.disabled = true;
                        submitBtn.textContent = '⏳ Processing...';
                        
                        // Convert to base64
                        const reader = new FileReader();
                        reader.onloadend = function() {{
                            const base64Audio = reader.result;
                            console.log('Converting audio to base64 for Streamlit...');
                            
                            // Store in localStorage
                            try {{
                                localStorage.setItem('streamlit_audio_' + key, base64Audio);
                                console.log('✅ Audio stored in localStorage');
                            }} catch(e) {{
                                console.log('localStorage not available');
                            }}
                            
                            // Find and update Streamlit input
                            const inputKey = 'audio_base64_' + key;
                            let attempts = 0;
                            const maxAttempts = 30;
                            
                            function updateInput() {{
                                attempts++;
                                console.log('Attempt ' + attempts + ': Looking for input ' + inputKey);
                                
                                if (attempts > maxAttempts) {{
                                    console.error('❌ Could not find input after ' + maxAttempts + ' attempts');
                                    alert('Could not send audio to Streamlit. Please refresh the page.');
                                    submitBtn.disabled = false;
                                    submitBtn.textContent = '📤 Submit Recording';
                                    return;
                                }}
                                
                                // Find input in parent document
                                let inputs = [];
                                if (window.parent && window.parent.document) {{
                                    inputs = window.parent.document.querySelectorAll('input[type="text"]');
                                }}
                                
                                console.log('Found ' + inputs.length + ' text inputs');
                                
                                for (let input of inputs) {{
                                    const inputId = (input.id || '').toLowerCase();
                                    const inputName = (input.name || '').toLowerCase();
                                    const inputValue = input.value || '';
                                    
                                    // Check if this is our input (empty or matches key)
                                    if (inputId.includes(inputKey.toLowerCase()) || 
                                        inputName.includes(inputKey.toLowerCase()) ||
                                        (inputValue === '' && attempts <= 5)) {{
                                        
                                        console.log('✅ FOUND INPUT! Setting value...');
                                        input.value = base64Audio;
                                        input.setAttribute('data-audio-submitted', 'true');
                                        
                                        // Trigger events
                                        ['input', 'change'].forEach(eventType => {{
                                            input.dispatchEvent(new Event(eventType, {{ bubbles: true }}));
                                        }});
                                        
                                        console.log('✅ Value set! Setting flag for Streamlit to process...');
                                        
                                        // Set a flag in the input that Streamlit can check
                                        input.setAttribute('data-submitted', 'true');
                                        
                                        // Show success message
                                        submitBtn.textContent = '✅ Submitted! Processing...';
                                        submitBtn.style.background = '#28a745';
                                        submitBtn.disabled = true;
                                        
                                        // Add a hidden button that Streamlit can auto-click to trigger rerun
                                        // We'll create a button in Streamlit that checks for this flag
                                        console.log('Audio submitted! Streamlit will process it on next rerun.');
                                        
                                        return;
                                    }}
                                }}
                                
                                // Retry
                                setTimeout(updateInput, 150);
                            }};
                            
                            updateInput();
                        }};
                        reader.readAsDataURL(audioBlob);
                    }});
                    
                    // Record again button
                    recordAgainBtn.addEventListener('click', function() {{
                        playbackDiv.style.display = 'none';
                        audioBlob = null;
                        audioChunks = [];
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
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
        
        # Hidden text input for JavaScript to populate with base64 audio
        # Make it easier for JavaScript to find by using a unique key
        audio_base64_key = f"audio_base64_{recording_key}"
        
        # Check if we should read from localStorage (set by JavaScript)
        # This allows JavaScript to set value without page reload
        if 'localStorage_audio_read' not in st.session_state:
            st.session_state['localStorage_audio_read'] = {}
        
        # Try to read from URL params or use session state
        audio_base64 = st.text_input(
            "Audio Data",
            key=audio_base64_key,
            label_visibility="collapsed",
            value=st.session_state.get(audio_base64_key, ""),
            help="Hidden input for audio data"
        )
        
        # Store in session state so it persists
        if audio_base64:
            st.session_state[audio_base64_key] = audio_base64
        
        # Auto-process when audio_base64 is received
        if audio_base64 and (audio_base64.startswith('data:audio') or audio_base64.startswith('data:application')):
            if not st.session_state.get(f'audio_stored_{recording_key}', False):
                try:
                    # Extract base64 part after comma
                    if ',' in audio_base64:
                        base64_data = audio_base64.split(',')[1]
                        mime_type = audio_base64.split(';')[0].split(':')[1] if ':' in audio_base64.split(';')[0] else 'audio/webm'
                    else:
                        base64_data = audio_base64
                        mime_type = 'audio/webm'
                    
                    audio_bytes = base64.b64decode(base64_data)
                    
                    # Convert webm to wav if needed
                    audio_format = 'wav'
                    if 'webm' in mime_type.lower():
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
                    
                    # Store audio for later processing
                    st.session_state[f'audio_bytes_{recording_key}'] = audio_bytes
                    st.session_state[f'audio_format_{recording_key}'] = audio_format
                    st.session_state[f'audio_stored_{recording_key}'] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error storing audio: {e}")
        
        # Auto-check button - checks if audio was submitted and triggers processing
        # This button auto-clicks if audio_base64 has data but isn't processed yet
        check_audio_key = f"check_audio_{recording_key}"
        if audio_base64 and (audio_base64.startswith('data:audio') or audio_base64.startswith('data:application')):
            if not st.session_state.get(f'audio_stored_{recording_key}', False):
                # Auto-process - no button needed, just process it
                pass  # Processing happens in the if block above
            else:
                # Already processed, show status
                st.info("✅ Audio already processed!")
        else:
            # No audio yet, show waiting message
            pass
        
        # Show playback if audio is stored (but not yet submitted)
        audio_stored = st.session_state.get(f'audio_stored_{recording_key}', False)
        audio_submitted = st.session_state.get(audio_submitted_key, False)
        
        # DEBUG: Show state - ALWAYS VISIBLE
        st.error("🔍🔍🔍 STATE DEBUG 🔍🔍🔍")
        st.write(f"**audio_stored:** {audio_stored}")
        st.write(f"**audio_submitted:** {audio_submitted}")
        st.write(f"**recording_key:** {recording_key}")
        st.write(f"**All session state keys containing 'recording':**")
        recording_keys = [k for k in st.session_state.keys() if 'recording' in k.lower()]
        st.write(recording_keys)
        
        if audio_stored and not audio_submitted:
            audio_bytes = st.session_state.get(f'audio_bytes_{recording_key}')
            if audio_bytes:
                st.success("✅ Recording completed! Listen to your recording:")
                st.audio(audio_bytes, format="audio/webm")
                
                # Submit button - matches Flask version (ONLY show if not submitted yet)
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📤 Submit Recording", type="primary", key=f"submit_{recording_key}", use_container_width=True):
                        # Process ASR when submit is clicked
                        with st.spinner("🔄 Processing audio with ASR..."):
                            audio_format = st.session_state.get(f'audio_format_{recording_key}', 'wav')
                            asr_transcript = call_sarvam_asr(
                                audio_bytes,
                                language,
                                st.secrets.get('SARVAM_API_KEY', os.environ.get('SARVAM_API_KEY', None)),
                                audio_format=audio_format
                            )
                            
                            if asr_transcript:
                                matches = check_match(crop_name, asr_transcript)
                                st.session_state[f'asr_result_{recording_key}'] = {
                                    'transcript': asr_transcript,
                                    'matches': matches
                                }
                            else:
                                st.session_state[f'asr_result_{recording_key}'] = {
                                    'transcript': None,
                                    'matches': False
                                }
                            
                            st.session_state[audio_submitted_key] = True
                            st.rerun()
                
                with col2:
                    if st.button("🔄 Record Again", key=f"rerecord_{recording_key}", use_container_width=True):
                        # Clear all state for this recording
                        for key in [f'audio_bytes_{recording_key}', f'audio_format_{recording_key}', 
                                   f'audio_stored_{recording_key}', f'asr_result_{recording_key}', 
                                   audio_submitted_key]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
        
        # Show results immediately after submit (matches Flask) - OUTSIDE the audio_stored block
        if st.session_state.get(audio_submitted_key, False):
            result = st.session_state.get(f'asr_result_{recording_key}', {})
            st.markdown("---")
            st.markdown("### ✅ Recording Submitted!")
            
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
                        "timestamp": datetime.now().isoformat(),
                        "audio_recorded": True
                    }
                    st.session_state.test_results.append(test_result)
                    st.session_state[result_saved_key] = True
            else:
                st.error("❌ ASR processing failed. Please check your API key.")
            
            # Show navigation based on attempt number (like Flask)
            st.markdown("---")
            
            # If more attempts remaining, show "Record Again" to continue with same crop
            if current_attempt_num < max_attempts:
                st.markdown(f"### Continue Testing ({current_attempt_num}/{max_attempts} attempts completed)")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Record Again", type="primary", key=f"next_attempt_{recording_key}", use_container_width=True):
                        # Move to next attempt for same crop
                        st.session_state.current_attempt[crop_index] = current_attempt_num + 1
                        # Clear this attempt's state
                        result_saved_key = f'result_saved_{recording_key}'
                        for key in [f'audio_bytes_{recording_key}', f'audio_format_{recording_key}', 
                                   f'audio_stored_{recording_key}', f'asr_result_{recording_key}', 
                                   audio_submitted_key, result_saved_key]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
                with col2:
                    if st.button("➡️ Next Crop", key=f"skip_to_next_crop_{recording_key}", use_container_width=True):
                        # Skip remaining attempts and move to next crop
                        st.session_state.current_test_index += 1
                        # Clear all state for this crop
                        for key in list(st.session_state.keys()):
                            if f'recording_{crop_index}' in key:
                                del st.session_state[key]
                        if crop_index in st.session_state.current_attempt:
                            del st.session_state.current_attempt[crop_index]
                        st.rerun()
            else:
                # All 5 attempts done, move to next crop
                st.markdown("### ✅ All 5 attempts completed!")
                st.info(f"You've completed all {max_attempts} attempts for {crop_name}. Moving to next crop...")
                col1, col2 = st.columns([2, 1])
                with col1:
                    if st.button("➡️ Next Crop", type="primary", key=f"next_crop_{recording_key}", use_container_width=True):
                        # Move to next crop
                        st.session_state.current_test_index += 1
                        # Clear all state for this crop
                        for key in list(st.session_state.keys()):
                            if f'recording_{crop_index}' in key:
                                del st.session_state[key]
                        if crop_index in st.session_state.current_attempt:
                            del st.session_state.current_attempt[crop_index]
                        st.rerun()
                with col2:
                    if st.button("🔄 Review Results", key=f"review_{recording_key}", use_container_width=True):
                        # Show results for this crop
                        st.rerun()
        
        # End Session button (matches Flask version)
        st.markdown("---")
        if st.button("🏁 End Session & Get Reports", type="primary"):
            st.session_state.current_test_index = len(st.session_state.test_data)
            st.rerun()
    
    else:
        # Testing complete
        st.header("🎉 Testing Complete!")
        
        # Results summary
        st.markdown(f"**Total Tests:** {len(st.session_state.test_results)}")
        if st.session_state.test_results:
            # Count matches (handle both old 'accuracy' format and new 'match' format)
            matches_count = sum(1 for r in st.session_state.test_results if r.get('match') == 'Yes' or (r.get('match') is True) or (r.get('accuracy', 0) > 0))
            st.markdown(f"**Matches:** {matches_count} / {len(st.session_state.test_results)}")
        
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
            
            st.download_button(
                label="📥 Download Results CSV",
                data=csv,
                file_name=filename,
                mime="text/csv",
                type="primary"
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
