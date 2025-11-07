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

# Sarvam ASR API Configuration
SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
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

def call_sarvam_asr(audio_bytes, language_code, api_key=None):
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
            api_key = st.secrets.get('SARVAM_API_KEY', os.environ.get('SARVAM_API_KEY', ''))
        
        if not api_key:
            st.warning("⚠️ Sarvam API key not configured. Using mock transcription.")
            return None
        
        # Get BCP47 language code
        bcp47_lang = BCP47_CODES.get(language_code.lower(), "hi-IN")
        
        # Prepare request
        files = {
            'file': ('audio.wav', audio_bytes, 'audio/wav')
        }
        
        data = {
            'model': 'saarika:v2.5',
            'language_code': bcp47_lang
        }
        
        headers = {
            'api-subscription-key': api_key
        }
        
        # Make API call
        response = requests.post(SARVAM_API_URL, files=files, data=data, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            transcript = result.get('transcript', result.get('text', '')).strip()
            return transcript
        else:
            st.error(f"Sarvam API error: {response.status_code}")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ ASR API call failed: {str(e)}. Using mock transcription.")
        return None

def calculate_accuracy(expected_text, actual_text):
    """
    Calculate accuracy by comparing expected crop name with ASR transcription
    
    Args:
        expected_text: Expected crop name (e.g., "Wheat")
        actual_text: ASR transcription result
    
    Returns:
        float: Accuracy percentage (0-100)
    """
    if not actual_text:
        return 0.0
    
    # Normalize both texts (lowercase, remove extra spaces)
    expected_clean = re.sub(r'[^\w\s]', '', expected_text.lower().strip())
    actual_clean = re.sub(r'[^\w\s]', '', actual_text.lower().strip())
    
    # Exact match
    if expected_clean == actual_clean:
        return 100.0
    
    # Check if expected word is in the transcription
    if expected_clean in actual_clean:
        return 95.0  # High accuracy if crop name found in transcription
    
    # Check if transcription contains key parts of expected word
    expected_words = expected_clean.split()
    actual_words = actual_clean.split()
    
    matches = sum(1 for word in expected_words if word in actual_clean)
    if matches > 0:
        # Partial match - calculate based on word matches
        return (matches / len(expected_words)) * 90.0
    
    # No match
    return 0.0

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
    """Main testing interface"""
    if st.session_state.current_test_index < len(st.session_state.test_data):
        current_crop = st.session_state.test_data[st.session_state.current_test_index]
        
        # Handle different column name formats
        crop_name = current_crop.get('crop_name') or current_crop.get('name', 'Unknown')
        crop_code = current_crop.get('crop_code') or current_crop.get('code', 'N/A')
        language = current_crop.get('language', st.session_state.selected_language or 'en')
        
        st.header(f"🎤 Testing: {crop_name}")
        st.markdown(f"**Crop Code:** {crop_code} | **Language:** {language.title() if isinstance(language, str) else language}")
        
        # Progress
        progress = (st.session_state.current_test_index + 1) / len(st.session_state.test_data)
        st.progress(progress)
        st.markdown(f"**Progress:** {st.session_state.current_test_index + 1} of {len(st.session_state.test_data)}")
        
        # Test sentence
        test_sentence = f"Please say {crop_name}"
        st.markdown(f"**Say this sentence:** \"{test_sentence}\"")
        
        # Audio Recording Component
        st.markdown("### 🎤 Record Audio")
        
        # Create unique key for this crop's recording
        recording_key = f"recording_{st.session_state.current_test_index}"
        
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
                <a id="downloadLink-{recording_key}" download="recording.wav" style="
                    display: inline-block;
                    background: #28a745;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    text-decoration: none;
                    margin: 10px;
                ">📥 Download Recording</a>
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
            const downloadLink = document.getElementById('downloadLink-' + key);
            
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
                        downloadLink.href = audioUrl;
                        playbackDiv.style.display = 'block';
                        
                        // Automatically convert to base64 and store for Streamlit
                        const reader = new FileReader();
                        reader.onloadend = function() {{
                            const base64Audio = reader.result;
                            
                            // Store in window object for Streamlit to access
                            window['audioData_' + key] = base64Audio;
                            
                            // Also store in sessionStorage as backup
                            try {{
                                sessionStorage.setItem('audio_' + key, base64Audio);
                            }} catch(e) {{
                                console.log('sessionStorage not available');
                            }}
                            
                            // Try to update Streamlit input via postMessage
                            if (window.parent && window.parent !== window) {{
                                window.parent.postMessage({{
                                    type: 'streamlit:setComponentValue',
                                    key: key,
                                    value: base64Audio
                                }}, '*');
                            }}
                            
                            // Also try direct DOM manipulation as fallback
                            setTimeout(function() {{
                                const inputs = window.parent.document.querySelectorAll('input[type="text"]');
                                for (let input of inputs) {{
                                    if (input.value === '' || input.getAttribute('data-base-input') === 'true') {{
                                        input.value = base64Audio;
                                        input.setAttribute('data-base-input', 'true');
                                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        break;
                                    }}
                                }}
                            }}, 100);
                        }};
                        reader.readAsDataURL(audioBlob);
                        
                        stream.getTracks().forEach(track => track.stop());
                    }};
                    
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
        
        # Add a hidden text input to receive base64 audio from JavaScript
        # This will be populated automatically when recording stops
        audio_base64 = st.text_input(
            "Audio Data (hidden)",
            key=f"audio_base64_{recording_key}",
            label_visibility="collapsed",
            value=""
        )
        
        # Convert base64 to bytes if audio was recorded and process ASR automatically
        asr_result_key = f"asr_result_{recording_key}"
        asr_processed_key = f"asr_processed_{recording_key}"
        
        if audio_base64 and (audio_base64.startswith('data:audio') or audio_base64.startswith('data:application')):
            try:
                # Extract base64 part after comma
                if ',' in audio_base64:
                    base64_data = audio_base64.split(',')[1]
                else:
                    base64_data = audio_base64
                audio_bytes = base64.b64decode(base64_data)
                st.session_state.recorded_audio = audio_bytes
                
                # Automatically process ASR if not already processed
                if not st.session_state.get(asr_processed_key, False):
                    with st.spinner("🔄 Processing audio with ASR..."):
                        asr_transcript = call_sarvam_asr(
                            audio_bytes,
                            language,
                            st.secrets.get('SARVAM_API_KEY', os.environ.get('SARVAM_API_KEY', None))
                        )
                        
                        if asr_transcript:
                            accuracy = calculate_accuracy(crop_name, asr_transcript)
                            st.session_state[asr_result_key] = {
                                'transcript': asr_transcript,
                                'accuracy': accuracy,
                                'matches': accuracy > 0
                            }
                        else:
                            st.session_state[asr_result_key] = {
                                'transcript': None,
                                'accuracy': 0.0,
                                'matches': False
                            }
                        
                        st.session_state[asr_processed_key] = True
                        st.rerun()
            except Exception as e:
                st.error(f"Error processing audio: {e}")
        
        # Display status and ASR results
        if st.session_state.recorded_audio:
            st.success("✅ Audio recorded!")
            st.markdown("### 🔊 Listen to Your Recording")
            st.audio(st.session_state.recorded_audio, format="audio/webm")
            
            # Show ASR results if available
            if st.session_state.get(asr_result_key):
                result = st.session_state[asr_result_key]
                st.markdown("---")
                st.markdown("### 🎯 ASR Results")
                
                if result['transcript']:
                    st.markdown(f"**Transcribed:** {result['transcript']}")
                    st.markdown(f"**Expected:** {crop_name}")
                    
                    if result['matches']:
                        if result['accuracy'] >= 95:
                            st.success(f"✅ **MATCH!** Accuracy: {result['accuracy']:.1f}%")
                        else:
                            st.warning(f"⚠️ **PARTIAL MATCH** Accuracy: {result['accuracy']:.1f}%")
                    else:
                        st.error(f"❌ **NO MATCH** Accuracy: {result['accuracy']:.1f}%")
                else:
                    st.error("❌ ASR processing failed. Please check your API key.")
            
            if st.button("🔄 Record Again", key=f"rerecord_{recording_key}"):
                st.session_state.recorded_audio = None
                st.session_state[asr_result_key] = None
                st.session_state[asr_processed_key] = False
                st.rerun()
        else:
            st.info("🎤 Click 'Start Recording' above. ASR results will appear automatically after you stop recording.")
        
        # Navigation
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous") and st.session_state.current_test_index > 0:
                st.session_state.current_test_index -= 1
                st.session_state.recorded_audio = None  # Clear audio when moving
                st.rerun()
        
        with col2:
            if st.button("Skip ➡️"):
                st.session_state.current_test_index += 1
                st.session_state.recorded_audio = None  # Clear audio when moving
                st.rerun()
        
        with col3:
            if st.button("✅ Save & Next"):
                # Get ASR results from automatic processing
                asr_result_data = st.session_state.get(asr_result_key, {})
                asr_transcript = asr_result_data.get('transcript', None)
                accuracy = asr_result_data.get('accuracy', 0.0)
                
                if asr_transcript:
                    asr_result = f"{asr_transcript} (accuracy: {accuracy:.1f}%)"
                elif st.session_state.recorded_audio:
                    asr_result = f"Recorded audio for {crop_name} (ASR processing failed)"
                    accuracy = 0.0
                else:
                    st.warning("⚠️ No audio recorded! Please record audio first.")
                    asr_result = f"Recorded audio for {crop_name} (no audio recorded)"
                    accuracy = 0.0
                
                result = {
                    "qa_name": st.session_state.qa_name,
                    "qa_email": st.session_state.user_info.get('email', '') if st.session_state.user_info else '',
                    "crop_name": crop_name,
                    "crop_code": crop_code,
                    "language": language,
                    "expected": crop_name,
                    "actual": asr_transcript if asr_transcript else asr_result,
                    "accuracy": accuracy,
                    "timestamp": datetime.now().isoformat(),
                    "audio_recorded": st.session_state.recorded_audio is not None
                }
                st.session_state.test_results.append(result)
                # Clear recorded audio and ASR results for next test
                st.session_state.recorded_audio = None
                st.session_state[asr_result_key] = None
                st.session_state[asr_processed_key] = False
                st.session_state.current_test_index += 1
                st.success("✅ Test saved! Moving to next crop...")
                st.rerun()
        
        with col4:
            if st.button("🏁 Finish Testing", type="primary"):
                # Get ASR results from automatic processing
                asr_result_data = st.session_state.get(asr_result_key, {})
                asr_transcript = asr_result_data.get('transcript', None)
                accuracy = asr_result_data.get('accuracy', 0.0)
                
                if asr_transcript:
                    asr_result = f"{asr_transcript} (accuracy: {accuracy:.1f}%)"
                elif st.session_state.recorded_audio:
                    asr_result = f"Recorded audio for {crop_name} (ASR processing failed)"
                    accuracy = 0.0
                else:
                    st.warning("⚠️ No audio recorded! Please record audio first.")
                    asr_result = f"Recorded audio for {crop_name} (no audio recorded)"
                    accuracy = 0.0
                
                result = {
                    "qa_name": st.session_state.qa_name,
                    "qa_email": st.session_state.user_info.get('email', '') if st.session_state.user_info else '',
                    "crop_name": crop_name,
                    "crop_code": crop_code,
                    "language": language,
                    "expected": crop_name,  # Store just the crop name, not the full sentence
                    "actual": asr_transcript if asr_transcript else asr_result,
                    "accuracy": accuracy,
                    "timestamp": datetime.now().isoformat(),
                    "audio_recorded": st.session_state.recorded_audio is not None
                }
                st.session_state.test_results.append(result)
                # Jump to end to show results
                st.session_state.current_test_index = len(st.session_state.test_data)
                st.session_state.recorded_audio = None
                st.session_state[asr_result_key] = None
                st.session_state[asr_processed_key] = False
                st.success("✅ Testing session completed! Showing results...")
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
