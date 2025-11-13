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
# Use same model name as Flask app (which uses default "saarika:v2.5")
MODEL_NAME = "saarika:v2.5"
BCP47_CODES = {
    # Short codes (Streamlit uses these)
    "en": "en-IN",
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "bn": "bn-IN",
    "gu": "gu-IN",
    "mr": "mr-IN",
    "pa": "pa-IN",
    # Full names (Flask uses these, CSV might have these)
    "english": "en-IN",
    "hindi": "hi-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "marathi": "mr-IN",
    "punjabi": "pa-IN",
    "odia": "or-IN"  # Also support odia
}

def call_sarvam_asr(audio_bytes, language_code, api_key=None, audio_format='wav', debug_expander=None):
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
        st.info(f"🔍 Calling ASR API:\n- URL: {SARVAM_API_URL}\n- Model: {MODEL_NAME}\n- Language: {bcp47_lang}\n- Format: {audio_format}\n- Audio size: {len(audio_bytes)} bytes\n- API Key: {'✅ Set' if api_key else '❌ Missing'}")
        
        if debug_expander:
            with debug_expander:
                st.write("**Step 5.1: Request Details**")
                st.code(f"URL: {SARVAM_API_URL}\nMethod: POST\nHeaders: api-subscription-key: {'Set' if api_key else 'Missing'}\nFiles: audio.{audio_format} ({len(audio_bytes)} bytes)\nData: model={MODEL_NAME}, language_code={bcp47_lang}", language='text')
        
        # Make API call
        response = None
        request_error = None
        try:
            if debug_expander:
                with debug_expander:
                    st.write("**Step 5.2: Sending HTTP Request**")
                    st.info("⏳ Sending POST request to API...")
            
            response = requests.post(SARVAM_API_URL, files=files, data=data, headers=headers, timeout=30)
            
            if debug_expander:
                with debug_expander:
                    st.write("**Step 5.3: Response Received**")
                    st.code(f"Status Code: {response.status_code}\nHeaders: {dict(response.headers)}\nContent-Type: {response.headers.get('Content-Type', 'N/A')}\nContent-Length: {len(response.content)} bytes", language='text')
        except requests.exceptions.Timeout as timeout_err:
            request_error = f"Timeout: {timeout_err}"
            st.error(f"❌ Request timed out after 30 seconds: {timeout_err}")
            if debug_expander:
                with debug_expander:
                    st.write("**Step 5.3: Request Failed (Timeout)**")
                    st.error(f"⏱️ Request timed out after 30 seconds\nError: {timeout_err}")
            # Store error in session state
            if hasattr(st, 'session_state'):
                st.session_state['_last_api_response'] = {
                    'status_code': 'TIMEOUT',
                    'error': 'Request timed out after 30 seconds',
                    'error_details': str(timeout_err),
                    'response': None
                }
            return None
        except requests.exceptions.ConnectionError as conn_err:
            request_error = f"Connection Error: {conn_err}"
            st.error(f"❌ Could not connect to API: {conn_err}")
            if debug_expander:
                with debug_expander:
                    st.write("**Step 5.3: Request Failed (Connection Error)**")
                    st.error(f"🔌 Connection failed\nError: {conn_err}\n\nPossible causes:\n- API server is down\n- Network issue\n- Firewall blocking connection")
            # Store error in session state
            if hasattr(st, 'session_state'):
                st.session_state['_last_api_response'] = {
                    'status_code': 'CONNECTION_ERROR',
                    'error': 'Could not connect to API server',
                    'error_details': str(conn_err),
                    'response': None
                }
            return None
        except Exception as req_error:
            request_error = f"Request Error: {req_error}"
            st.error(f"❌ Request failed: {req_error}")
            if debug_expander:
                with debug_expander:
                    st.write("**Step 5.3: Request Failed**")
                    st.error(f"❌ Request failed\nError: {req_error}\nError Type: {type(req_error).__name__}")
            # Store error in session state
            if hasattr(st, 'session_state'):
                st.session_state['_last_api_response'] = {
                    'status_code': 'ERROR',
                    'error': f'Request failed: {str(req_error)}',
                    'error_details': str(req_error),
                    'error_type': type(req_error).__name__,
                    'response': None
                }
            return None
        
        # If we got here, we have a response - store it immediately
        if response is not None and hasattr(st, 'session_state'):
            # Store response immediately (even before parsing)
            try:
                response_text = response.text[:5000]  # Store first 5000 chars
                st.session_state['_last_api_response'] = {
                    'status_code': response.status_code,
                    'headers': dict(response.headers),
                    'content_type': response.headers.get('Content-Type', 'N/A'),
                    'content_length': len(response.content),
                    'raw_response_preview': response_text,
                    'response': None  # Will be filled after JSON parsing
                }
            except:
                pass  # If storing fails, continue anyway
        
        if response.status_code == 200:
            try:
                result = response.json()
                
                # Update session state with parsed response
                if hasattr(st, 'session_state') and st.session_state.get('_last_api_response'):
                    st.session_state['_last_api_response']['response'] = result
                    st.session_state['_last_api_response']['available_fields'] = list(result.keys()) if isinstance(result, dict) else 'Not a dict'
                
                if debug_expander:
                    with debug_expander:
                        st.write("**Step 5.4: Parsing JSON Response**")
                        st.success("✅ Response is valid JSON")
                        st.json(result)
                        st.write("**Step 5.5: Checking for Transcript**")
                        st.code(f"Available fields: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", language='text')
                
                transcript = result.get('transcript', result.get('text', '')).strip()
                
                if debug_expander:
                    with debug_expander:
                        if transcript:
                            st.success(f"✅ Transcript found in standard fields: '{transcript}'")
                        else:
                            st.warning("⚠️ No transcript in 'transcript' or 'text' fields")
                if transcript:
                    st.success(f"✅ ASR Success: '{transcript}'")
                    return transcript
                else:
                    # API returned 200 but no transcript - DEBUG HEAVILY
                    import json
                    
                    # Show what fields ARE in the response (IMMEDIATELY VISIBLE)
                    available_fields = list(result.keys()) if isinstance(result, dict) else 'Not a dict'
                    
                    # Show full response IMMEDIATELY (not hidden) - SUPER PROMINENT
                    st.error(f"❌ API returned 200 but no transcript in response")
                    st.markdown("---")
                    st.markdown("### 🔍 **DEBUG INFO (IMMEDIATELY VISIBLE)**")
                    st.markdown("**This info will also appear in Error Logs section below**")
                    
                    # Show full response in multiple formats
                    st.write("**📋 Full API Response (JSON):**")
                    st.json(result)
                    
                    # Also show as code block
                    st.write("**📋 Full API Response (Formatted):**")
                    st.code(json.dumps(result, indent=2), language='json')
                    
                    st.warning(f"📋 **Available fields in response:** `{available_fields}`")
                    
                    # Try to find transcript in other possible field names (check nested too)
                    possible_transcript_fields = ['transcript', 'text', 'transcription', 'result', 'output', 'data', 'content', 'message', 'transcribed_text', 'asr_result']
                    found_transcript = None
                    found_field = None
                    
                    # Check top-level fields
                    for field in possible_transcript_fields:
                        if field in result and result[field]:
                            value = result[field]
                            # If it's a string, use it
                            if isinstance(value, str) and value.strip():
                                found_transcript = value
                                found_field = field
                                break
                            # If it's a dict, check nested fields
                            elif isinstance(value, dict):
                                for nested_field in possible_transcript_fields:
                                    if nested_field in value and value[nested_field]:
                                        found_transcript = value[nested_field]
                                        found_field = f"{field}.{nested_field}"
                                        break
                                if found_transcript:
                                    break
                    
                    # Also check ALL string values in the response
                    if not found_transcript:
                        def find_text_in_dict(d, path=""):
                            """Recursively find any string values that might be transcript"""
                            for k, v in d.items():
                                current_path = f"{path}.{k}" if path else k
                                if isinstance(v, str) and len(v.strip()) > 0:
                                    # If it looks like a transcript (has letters/words)
                                    if any(c.isalpha() for c in v) and len(v.strip()) > 2:
                                        return v, current_path
                                elif isinstance(v, dict):
                                    result = find_text_in_dict(v, current_path)
                                    if result:
                                        return result
                            return None
                        
                        text_result = find_text_in_dict(result)
                        if text_result:
                            found_transcript, found_field = text_result
                    
                    if found_transcript:
                        st.success(f"✅ **Found transcript in field `{found_field}`:** `{found_transcript}`")
                        transcript = str(found_transcript).strip()
                        if transcript:
                            st.balloons()
                            return transcript
                    else:
                        st.error("❌ **No transcript found in any field!** Check the JSON above.")
                    
                    # Store in session state for error logs
                    if hasattr(st, 'session_state'):
                        st.session_state['_last_api_response'] = {
                            'status_code': 200,
                            'response': result,
                            'available_fields': available_fields,
                            'error': 'No transcript in response'
                        }
                    
                    return None
            except Exception as json_error:
                st.error(f"❌ Failed to parse API response: {json_error}")
                st.code(f"Raw response: {response.text[:1000]}", language='text')
                # Store raw response for debugging
                if hasattr(st, 'session_state'):
                    st.session_state['_last_api_response'] = {
                        'status_code': 200,
                        'raw_response': response.text[:2000],
                        'error': f'JSON parse error: {json_error}',
                        'response': None
                    }
                if debug_expander:
                    with debug_expander:
                        st.write("**Step 5.4: JSON Parse Error**")
                        st.error(f"❌ Failed to parse JSON\nError: {json_error}\nRaw response (first 1000 chars):\n{response.text[:1000]}")
                return None
        else:
            st.error(f"❌ API error: HTTP {response.status_code}")
            try:
                error_body = response.json()
                st.error(f"Error details: {error_body}")
                st.code(f"Full error response: {error_body}", language='json')
                # Store error response for debugging
                if hasattr(st, 'session_state'):
                    st.session_state['_last_api_response'] = {
                        'status_code': response.status_code,
                        'response': error_body,
                        'error': f'HTTP {response.status_code}',
                        'available_fields': list(error_body.keys()) if isinstance(error_body, dict) else 'Not a dict'
                    }
                if debug_expander:
                    with debug_expander:
                        st.write("**Step 5.3: Non-200 Response**")
                        st.error(f"❌ API returned HTTP {response.status_code}\nResponse: {error_body}")
            except:
                error_text = response.text[:1000]
                st.error(f"Error text: {error_text}")
                st.code(f"Raw error response: {error_text}", language='text')
                # Store raw error for debugging
                if hasattr(st, 'session_state'):
                    st.session_state['_last_api_response'] = {
                        'status_code': response.status_code,
                        'raw_response': error_text,
                        'error': f'HTTP {response.status_code}',
                        'response': None
                    }
                if debug_expander:
                    with debug_expander:
                        st.write("**Step 5.3: Non-200 Response (Non-JSON)**")
                        st.error(f"❌ API returned HTTP {response.status_code}\nRaw response: {error_text}")
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
    """Step 2: CSV Upload - Matches Flask app behavior (simple first column)"""
    st.header("📁 Step 2: Upload CSV or Start Testing")
    
    st.markdown("Upload a CSV file with crop names. **Simple format:** Just put crop names in the first column (one per row).")
    st.info("💡 **Tip:** The Flask version just reads the first column - no specific format needed! You can have other columns too, we'll just use the first one.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV - match Flask behavior: just read first column
            import csv
            import io
            
            # Read as text first (like Flask does)
            content = uploaded_file.read().decode('utf-8')
            csvfile = io.StringIO(content)
            reader = csv.reader(csvfile)
            
            crop_names = []
            for row in reader:
                if row and row[0].strip():  # Skip empty rows, get first column
                    crop_names.append(row[0].strip())
            
            if crop_names:
                st.success(f"✅ Found {len(crop_names)} crop names!")
                
                # Show preview
                preview_df = pd.DataFrame({
                    'Crop Name': crop_names[:10]  # Show first 10
                })
                if len(crop_names) > 10:
                    st.dataframe(preview_df)
                    st.caption(f"... and {len(crop_names) - 10} more crops")
                else:
                    st.dataframe(preview_df)
                
                if st.button("Start Testing", type="primary"):
                    # Convert to test_data format (matching what the testing interface expects)
                    test_data = []
                    for idx, crop_name in enumerate(crop_names):
                        test_data.append({
                            'crop_name': crop_name,
                            'crop_code': f'CROP{idx+1:03d}',  # Auto-generate code
                            'language': st.session_state.selected_language or 'en',
                            'serial_number': idx + 1
                        })
                    st.session_state.test_data = test_data
                    st.rerun()
            else:
                st.error("No crop names found in CSV file. Make sure the first column has crop names.")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")
            st.code(str(e), language='text')
    
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
        
        # CRITICAL: Define audio_base64_key BEFORE the form so it's always available
        audio_base64_key = f"audio_base64_{recording_key}"
        
        # Audio Recording Component - COMPLETE REWRITE
        st.markdown("### 🎤 Record Audio")
        
        # Use Streamlit form for reliable submission
        with st.form(key=f"audio_form_{recording_key}", clear_on_submit=False):
            # Hidden input for base64 audio (populated by JavaScript)
            # Put it FIRST so it's rendered before the recorder
            audio_base64_data = st.text_input(
                "Audio Data",
                key=audio_base64_key,
                value="",
                label_visibility="collapsed",
                help="Hidden input for audio data"
            )
            
            # Custom Audio Recorder HTML Component
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
                        
                        // CRITICAL: Populate the input IMMEDIATELY when recording stops
                        // This ensures the input is ready when user clicks form submit button
                        try {{
                            const reader = new FileReader();
                            const base64Promise = new Promise((resolve, reject) => {{
                                reader.onload = () => {{
                                    const base64 = reader.result.split(',')[1];
                                    resolve(base64);
                                }};
                                reader.onerror = reject;
                            }});
                            
                            reader.readAsDataURL(audioBlob);
                            const base64Audio = await base64Promise;
                            
                            // Store in sessionStorage immediately
                            sessionStorage.setItem('audio_' + key, base64Audio);
                            
                            // Find and populate the hidden text input
                            let targetInput = null;
                            
                            // Strategy 1: Find by form context
                            const form = document.querySelector('form[data-testid*="stForm"]');
                            if (form) {{
                                const inputs = form.querySelectorAll('input[type="text"]');
                                for (let input of inputs) {{
                                    const container = input.closest('[data-testid*="stTextInput"]');
                                    if (container) {{
                                        const label = container.querySelector('label');
                                        if (!label || label.style.display === 'none' || label.textContent === '' || label.textContent === 'Audio Data') {{
                                            targetInput = input;
                                            break;
                                        }}
                                    }}
                                }}
                            }}
                            
                            // Strategy 2: Find any hidden text input
                            if (!targetInput) {{
                                const allInputs = Array.from(document.querySelectorAll('input[type="text"]'));
                                for (let input of allInputs) {{
                                    const container = input.closest('[data-testid*="stTextInput"]');
                                    if (container) {{
                                        const label = container.querySelector('label');
                                        const isHidden = !label || label.style.display === 'none' || 
                                                       label.textContent === '' || 
                                                       label.textContent === 'Audio Data';
                                        if (isHidden) {{
                                            targetInput = input;
                                            break;
                                        }}
                                    }}
                                }}
                            }}
                            
                            if (targetInput) {{
                                targetInput.value = base64Audio;
                                targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                console.log('Audio data populated in input immediately after recording');
                            }} else {{
                                console.warn('Could not find target input to populate');
                            }}
                        }} catch (error) {{
                            console.error('Error populating input after recording:', error);
                        }}
                        
                        stream.getTracks().forEach(track => track.stop());
                    }};
                    
                    // Submit button handler - Populate input and let form handle submission
                    submitBtn.addEventListener('click', async function() {{
                        if (!audioBlob) {{
                            alert('No recording to submit');
                            return;
                        }}
                        
                        submitBtn.disabled = true;
                        submitBtn.textContent = '⏳ Preparing...';
                        
                        try {{
                            // Convert blob to base64
                            const reader = new FileReader();
                            const base64Promise = new Promise((resolve, reject) => {{
                                reader.onload = () => {{
                                    const base64 = reader.result.split(',')[1];
                                    resolve(base64);
                                }};
                                reader.onerror = reject;
                            }});
                            
                            reader.readAsDataURL(audioBlob);
                            const base64Audio = await base64Promise;
                            
                            // Find and populate the hidden text input - MULTIPLE STRATEGIES
                            let targetInput = null;
                            
                            // Strategy 1: Find by data attribute
                            targetInput = document.querySelector('input[data-audio-input="' + key + '"]');
                            
                            // Strategy 2: Find by key in form
                            if (!targetInput) {{
                                const form = document.querySelector('form[data-testid*="stForm"]');
                                if (form) {{
                                    const inputs = form.querySelectorAll('input[type="text"]');
                                    for (let input of inputs) {{
                                        const container = input.closest('[data-testid*="stTextInput"]');
                                        if (container) {{
                                            const label = container.querySelector('label');
                                            if (!label || label.style.display === 'none' || label.textContent === '' || label.textContent === 'Audio Data') {{
                                                targetInput = input;
                                                break;
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                            
                            // Strategy 3: Find any hidden text input
                            if (!targetInput) {{
                                const allInputs = Array.from(document.querySelectorAll('input[type="text"]'));
                                for (let input of allInputs) {{
                                    const container = input.closest('[data-testid*="stTextInput"]');
                                    if (container) {{
                                        const label = container.querySelector('label');
                                        const isHidden = !label || label.style.display === 'none' || 
                                                       label.textContent === '' || 
                                                       label.textContent === 'Audio Data' ||
                                                       container.style.display === 'none';
                                        if (isHidden && input.value === '') {{
                                            targetInput = input;
                                            break;
                                        }}
                                    }}
                                }}
                            }}
                            
                            if (targetInput) {{
                                targetInput.value = base64Audio;
                                targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                
                                // Also store in sessionStorage as backup
                                sessionStorage.setItem('audio_' + key, base64Audio);
                                
                                // Find and click the form submit button
                                setTimeout(function() {{
                                    // Look for the form submit button
                                    const form = targetInput.closest('form');
                                    if (form) {{
                                        const submitBtn = form.querySelector('button[type="submit"]');
                                        if (submitBtn) {{
                                            submitBtn.click();
                                            return;
                                        }}
                                    }}
                                    
                                    // Fallback: find any button with Submit/Process text
                                    const submitButtons = Array.from(document.querySelectorAll('button'));
                                    const formSubmitBtn = submitButtons.find(btn => 
                                        (btn.textContent.includes('Submit') || btn.textContent.includes('Process')) &&
                                        btn.type === 'submit'
                                    );
                                    if (formSubmitBtn) {{
                                        formSubmitBtn.click();
                                    }} else {{
                                        submitBtn.textContent = '✅ Audio ready! Click "Submit & Process" below';
                                        submitBtn.style.background = '#28a745';
                                    }}
                                }}, 300);
                            }} else {{
                                // Last resort: store in sessionStorage and show message
                                sessionStorage.setItem('audio_' + key, base64Audio);
                                submitBtn.textContent = '✅ Audio ready! Click "Submit & Process" below';
                                submitBtn.style.background = '#28a745';
                                alert('Audio prepared! Please click the "Submit & Process" button below to process.');
                            }}
                            
                            submitBtn.textContent = '✅ Ready - Click Submit & Process below';
                            submitBtn.style.background = '#28a745';
                        }} catch (error) {{
                            console.error('Error preparing audio:', error);
                            alert('Error preparing audio: ' + error.message);
                            submitBtn.disabled = false;
                            submitBtn.textContent = '📤 Submit Recording';
                            submitBtn.style.background = '#007bff';
                        }}
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
            
            # Submit button
            submitted = st.form_submit_button("📤 Submit & Process", type="primary", use_container_width=True)
            
            if submitted:
                # CRITICAL: Read from session state first (where Streamlit stores widget values)
                audio_data = st.session_state.get(audio_base64_key, '')
                
                # Also try the widget value directly as fallback
                if (not audio_data or len(audio_data) < 100) and audio_base64_data:
                    audio_data = audio_base64_data
                
                # If still empty, try to read from sessionStorage via JavaScript injection
                if not audio_data or len(audio_data) < 100:
                    # Inject JS to read from sessionStorage and store in session state
                    read_storage_js = f"""
                    <script>
                    (function() {{
                        const audioData = sessionStorage.getItem('audio_{recording_key}');
                        if (audioData && audioData.length > 100) {{
                            // Store in Streamlit session state via the input
                            const input = document.querySelector('input[type="text"][data-baseweb*="input"]');
                            if (input) {{
                                input.value = audioData;
                                input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            }}
                            // Also try to find by key attribute
                            const inputs = Array.from(document.querySelectorAll('input[type="text"]'));
                            for (let inp of inputs) {{
                                const container = inp.closest('[data-testid*="stTextInput"]');
                                if (container) {{
                                    const label = container.querySelector('label');
                                    if (!label || label.style.display === 'none' || label.textContent === 'Audio Data') {{
                                        inp.value = audioData;
                                        inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                                        inp.dispatchEvent(new Event('change', {{ bubbles: true }}));
                                        break;
                                    }}
                                }}
                            }}
                            // Trigger rerun to process
                            setTimeout(function() {{
                                if (window.parent && window.parent.streamlit) {{
                                    window.parent.streamlit.rerun();
                                }} else {{
                                    window.location.reload();
                                }}
                            }}, 100);
                        }}
                    }})();
                    </script>
                    """
                    components.html(read_storage_js, height=0)
                    # Re-read from session state after JS injection
                    audio_data = st.session_state.get(audio_base64_key, '')
                
                # Process audio if we have it
                if audio_data and len(audio_data) > 100:
                    try:
                        import base64
                        audio_bytes = base64.b64decode(audio_data)
                        
                        # Convert webm to wav if needed
                        audio_format = 'webm'
                        try:
                            from pydub import AudioSegment
                            import io
                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                            wav_buffer = io.BytesIO()
                            audio_segment.export(wav_buffer, format="wav")
                            audio_bytes = wav_buffer.getvalue()
                            audio_format = 'wav'
                        except:
                            pass
                        
                        # Create mock file object
                        uploaded_audio = type('obj', (object,), {
                            'read': lambda: audio_bytes,
                            'name': f'recording_{recording_key}.webm',
                            'type': 'audio/webm'
                        })()
                        
                        # Process immediately
                        if not st.session_state.get(f'audio_processed_{recording_key}', False):
                            with st.spinner("🔄 Processing audio with ASR..."):
                                try:
                                    # Get API key
                                    api_key = None
                                    try:
                                        api_key = st.secrets.get('SARVAM_API_KEY', '')
                                    except:
                                        try:
                                            api_key = st.secrets.secrets.get('SARVAM_API_KEY', '')
                                        except:
                                            api_key = os.environ.get('SARVAM_API_KEY', '')
                                    
                                    # Call ASR API
                                    asr_transcript = call_sarvam_asr(
                                        audio_bytes,
                                        language,
                                        api_key,
                                        audio_format=audio_format,
                                        debug_expander=None
                                    )
                                    
                                    if asr_transcript:
                                        matches = check_match(crop_name, asr_transcript)
                                        st.session_state[f'asr_result_{recording_key}'] = {
                                            'transcript': asr_transcript,
                                            'matches': matches
                                        }
                                        st.session_state[audio_submitted_key] = True
                                        st.session_state[f'audio_processed_{recording_key}'] = True
                                        
                                        # Save result
                                        test_result = {
                                            "qa_name": st.session_state.qa_name,
                                            "qa_email": st.session_state.user_info.get('email', '') if st.session_state.user_info else '',
                                            "session_id": st.session_state.session_id,
                                            "crop_name": crop_name,
                                            "crop_code": crop_code,
                                            "language": language,
                                            "attempt_number": current_attempt_num,
                                            "expected": crop_name,
                                            "transcript": asr_transcript,
                                            "keyword_detected": "Yes" if matches else "No",
                                            "match": "Yes" if matches else "No",
                                            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                            "audio_recorded": True
                                        }
                                        st.session_state.test_results.append(test_result)
                                        
                                        # Save to Azure
                                        if AZURE_AVAILABLE:
                                            try:
                                                user_email = st.session_state.user_info.get('email', 'unknown@example.com') if st.session_state.user_info else 'unknown@example.com'
                                                upload_single_test_result(
                                                    test_result=test_result,
                                                    user_email=user_email,
                                                    language=language,
                                                    session_id=st.session_state.session_id
                                                )
                                            except:
                                                pass
                                        
                                        st.rerun()
                                    else:
                                        st.error("ASR API returned no transcript")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                                    st.code(str(e), language='text')
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
                        st.code(str(e), language='text')
                else:
                    st.warning("⚠️ No audio data found. Please record audio first.")
        
        # Fallback: File uploader (outside form)
        
        # Check if audio was submitted (via URL parameter)
        audio_submit_key = st.query_params.get('audio_submit')
        
        # If audio was submitted, read from sessionStorage and process IMMEDIATELY
        audio_bytes = None
        uploaded_audio = None
        
        # Check for audio_loaded param FIRST (handles second rerun after JS populated input)
        audio_loaded_key = st.query_params.get('audio_loaded')
        if audio_loaded_key == recording_key:
            if f'audio_loaded_{recording_key}' not in st.session_state:
                st.session_state[f'audio_loaded_{recording_key}'] = True
                # Clear the audio_submit flag to avoid re-entering first branch
                if 'audio_submit' in st.query_params:
                    st.query_params.pop('audio_submit', None)
                st.rerun()
        
        # If audio_loaded flag is set, audio should be in input - process it
        if f'audio_loaded_{recording_key}' in st.session_state:
            # Read from the form input (already created above) - DON'T create duplicate
            audio_base64_data = st.session_state.get(audio_base64_key, '')
            
            # Store in session state if we have data
            if audio_base64_data and len(audio_base64_data) > 100:
                st.session_state[audio_base64_key] = audio_base64_data
            
            # Process if we have audio data
            if audio_base64_data and len(audio_base64_data) > 100:
                try:
                    import base64
                    audio_bytes = base64.b64decode(audio_base64_data)
                    uploaded_audio = type('obj', (object,), {
                        'read': lambda: audio_bytes,
                        'name': f'recording_{recording_key}.webm',
                        'type': 'audio/webm'
                    })()
                    st.success("✅ Audio received! Processing...")
                    # Clear sessionStorage
                    clear_storage_js = f"""
                    <script>
                    sessionStorage.removeItem('audio_{recording_key}');
                    sessionStorage.removeItem('audio_format_{recording_key}');
                    sessionStorage.removeItem('audio_submit_{recording_key}');
                    </script>
                    """
                    components.html(clear_storage_js, height=0)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    st.code(str(e), language='text')
                    audio_bytes = None
        
        # If audio was just submitted (first time), inject JS to load from sessionStorage
        elif audio_submit_key == recording_key:
            # First time - inject JS to read sessionStorage and populate input, then rerun
            audio_loader_js = f"""
            <script>
            (function() {{
                const key = '{recording_key}';
                const audioBase64 = sessionStorage.getItem('audio_' + key);
                
                if (audioBase64) {{
                    console.log('Found audio in sessionStorage, length:', audioBase64.length);
                    
                    // Find the text input by its key attribute or position
                    setTimeout(function() {{
                        // Get all text inputs
                        const inputs = Array.from(document.querySelectorAll('input[type="text"]'));
                        
                        // Find the one that's for audio (should be the first hidden one)
                        let targetInput = null;
                        for (let input of inputs) {{
                            const container = input.closest('[data-testid*="stTextInput"]');
                            if (container) {{
                                // Check if it's hidden (label visibility collapsed)
                                const label = container.querySelector('label');
                                if (!label || label.style.display === 'none' || label.textContent === '') {{
                                    targetInput = input;
                                    break;
                                }}
                            }}
                        }}
                        
                        if (targetInput) {{
                            console.log('Found target input, setting value');
                            targetInput.value = audioBase64;
                            targetInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            targetInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
                            
                            // Trigger Streamlit rerun to process the audio
                            setTimeout(function() {{
                                // Use Streamlit's rerun mechanism
                                if (window.parent && window.parent.streamlit) {{
                                    try {{
                                        window.parent.streamlit.rerun();
                                    }} catch(e) {{
                                        // Fallback: reload page with flag
                                        const url = new URL(window.location.href);
                                        url.searchParams.set('audio_loaded', key);
                                        url.searchParams.set('_t', Date.now());
                                        window.parent.location.href = url.toString();
                                    }}
                                }} else {{
                                    const url = new URL(window.location.href);
                                    url.searchParams.set('audio_loaded', key);
                                    url.searchParams.set('_t', Date.now());
                                    window.location.href = url.toString();
                                }}
                            }}, 200);
                        }} else {{
                            console.error('Could not find target input');
                        }}
                    }}, 100);
                }} else {{
                    console.error('No audio found in sessionStorage');
                }}
            }})();
            </script>
            """
            components.html(audio_loader_js, height=0)
            st.info("⏳ Loading audio from sessionStorage...")
            # Don't create duplicate input - the form input above already exists
            # JS will populate it, then we'll read from session state on next rerun
        
        # If we don't have audio yet, read from form input (already created above)
        elif audio_bytes is None:
            # Read from the form input - DON'T create duplicate
            audio_base64_data = st.session_state.get(audio_base64_key, '')
        
        # Fallback: File uploader (only if direct processing failed)
        if not audio_bytes:
            uploaded_audio = st.file_uploader(
                "Upload audio (fallback only)",
                type=['webm', 'wav', 'mp3'],
                key=f"audio_upload_{recording_key}",
                help="Only use if direct processing doesn't work"
            )
            
            if uploaded_audio is not None:
                audio_bytes = uploaded_audio.read()
                st.success("✅ Audio file received!")
        
        # Process audio when uploaded (EXACT FLASK WORKFLOW)
        if audio_bytes and not st.session_state.get(f'audio_processed_{recording_key}', False):
            # COMPREHENSIVE DEBUG PANEL - ALWAYS VISIBLE
            st.markdown("---")
            st.markdown("### 🔍 **COMPREHENSIVE DEBUG PANEL**")
            debug_expander = st.expander("📊 **Step-by-Step Debug Info**", expanded=True)
            
            with debug_expander:
                st.write("**Step 1: Audio Upload Check**")
                st.code(f"Audio file: {uploaded_audio.name}\nSize: {len(audio_bytes)} bytes\nType: {uploaded_audio.type}", language='text')
            
            with st.spinner("🔄 Processing audio with ASR..."):
                try:
                    # Determine audio format
                    audio_format = 'wav'
                    conversion_status = "Not needed (already WAV)"
                    
                    with debug_expander:
                        st.write("**Step 2: Audio Format Detection**")
                        st.code(f"File extension: {uploaded_audio.name.split('.')[-1]}\nInitial format: {audio_format}", language='text')
                    
                    if uploaded_audio.name.endswith('.webm'):
                        # Try to convert to WAV if pydub is available (better compatibility)
                        # But if not available, use webm directly (API should accept it)
                        try:
                            with debug_expander:
                                st.write("**Step 2.1: Attempting WebM → WAV Conversion**")
                            from pydub import AudioSegment
                            import io
                            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="webm")
                            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
                            wav_buffer = io.BytesIO()
                            audio_segment.export(wav_buffer, format="wav")
                            audio_bytes = wav_buffer.getvalue()
                            audio_format = 'wav'
                            conversion_status = "✅ Successfully converted webm → wav"
                            st.success("✅ Converted webm to wav successfully!")
                            with debug_expander:
                                st.success(f"✅ Conversion successful!\nOriginal size: {len(audio_bytes)} bytes\nNew size: {len(audio_bytes)} bytes")
                        except (ImportError, Exception) as e:
                            # pydub not available or conversion failed - use webm directly
                            # API should accept webm format
                            audio_format = 'webm'
                            conversion_status = f"ℹ️ Using webm format directly (API accepts webm)\nReason: {str(e)}"
                            st.info("ℹ️ Using webm format directly - API will process it")
                            with debug_expander:
                                st.warning(f"⚠️ Conversion skipped\nReason: {str(e)}\nUsing original webm format")
                    
                    # Store conversion status for debugging
                    st.session_state[f'_conversion_status_{recording_key}'] = conversion_status
                    
                    # Store audio format for error logs
                    st.session_state[f'_audio_format_{recording_key}'] = audio_format
                    
                    with debug_expander:
                        st.write("**Step 3: Final Audio Format**")
                        st.code(f"Format: {audio_format}\nSize: {len(audio_bytes)} bytes\nConversion: {conversion_status}", language='text')
                    
                    # Call ASR API (like Flask)
                    # Get API key - try multiple methods (same as error logs section)
                    api_key = None
                    try:
                        api_key = st.secrets.get('SARVAM_API_KEY', '')
                        if not api_key or api_key == '':
                            try:
                                api_key = st.secrets.secrets.get('SARVAM_API_KEY', '')
                            except:
                                api_key = os.environ.get('SARVAM_API_KEY', '')
                    except Exception as e:
                        try:
                            api_key = st.secrets.secrets.get('SARVAM_API_KEY', '')
                        except:
                            api_key = os.environ.get('SARVAM_API_KEY', '')
                    
                    with debug_expander:
                        st.write("**Step 4: API Key Retrieval**")
                        st.code(f"API Key found: {'✅ Yes' if api_key else '❌ No'}\nKey preview: {api_key[:10] + '...' + api_key[-5:] if api_key else 'N/A'}", language='text')
                    
                    # Always call ASR - let it handle the error if key is missing
                    # This way we get the actual API error, not just "key not configured"
                    with debug_expander:
                        st.write("**Step 5: Calling ASR API**")
                        st.code(f"URL: {SARVAM_API_URL}\nModel: {MODEL_NAME}\nLanguage: {language} ({BCP47_CODES.get(language.lower(), 'hi-IN')})\nFormat: {audio_format}\nAudio size: {len(audio_bytes)} bytes", language='text')
                    
                    asr_transcript = call_sarvam_asr(
                        audio_bytes,
                        language,
                        api_key,  # Pass the key (or None if not found)
                        audio_format=audio_format,
                        debug_expander=debug_expander  # Pass debug expander for detailed logging
                    )
                    
                    if asr_transcript:
                        # Use Flask's exact keyword matching logic
                        matches = check_match(crop_name, asr_transcript)
                        st.session_state[f'asr_result_{recording_key}'] = {
                            'transcript': asr_transcript,
                            'matches': matches
                        }
                    else:
                        # ASR failed - call_sarvam_asr will have shown the actual error
                        st.session_state[f'asr_result_{recording_key}'] = {
                            'transcript': None,
                            'matches': False,
                            'error': 'ASR API returned no transcript - check error messages above'
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
                        conversion_status = st.session_state.get(f'_conversion_status_{recording_key}', 'Unknown')
                        st.code(f"Name: {uploaded_audio.name}\nSize: {audio_size} bytes\nType: {uploaded_audio.type}\nConversion: {conversion_status}", language='text')
                    
                    # Show what was actually sent to API
                    st.write("**Request Details:**")
                    # Get audio_format from session state if available
                    audio_format_used = st.session_state.get(f'_audio_format_{recording_key}', 'webm')
                    st.code(f"Headers: api-subscription-key: {'Set' if api_key_check != 'NOT FOUND' else 'Missing'}\nFiles: audio.{audio_format_used}\nData: model={MODEL_NAME}, language_code={BCP47_CODES.get(language.lower(), 'hi-IN')}", language='text')
                    
                    # Show actual API response if available - MAKE IT SUPER VISIBLE
                    st.markdown("---")
                    st.markdown("### 🔍 **ACTUAL API RESPONSE (CRITICAL DEBUG INFO)**")
                    
                    if st.session_state.get('_last_api_response'):
                        api_resp = st.session_state['_last_api_response']
                        import json
                        response_data = api_resp.get('response', api_resp.get('raw_response', 'N/A'))
                        
                        # Show status and fields first
                        st.write(f"**Status Code:** `{api_resp.get('status_code', 'N/A')}`")
                        st.write(f"**Error:** `{api_resp.get('error', 'N/A')}`")
                        available_fields = api_resp.get('available_fields', 'N/A')
                        st.write(f"**Available Fields:** `{available_fields}`")
                        
                        # Show full response in expandable JSON viewer (ALWAYS EXPANDED)
                        st.write("**Full Response JSON:**")
                        if isinstance(response_data, dict):
                            st.json(response_data)
                        else:
                            st.code(str(response_data), language='json')
                        
                        # Also show as formatted text for easy reading
                        st.write("**Formatted Response:**")
                        if isinstance(response_data, dict):
                            response_str = json.dumps(response_data, indent=2)
                        elif isinstance(response_data, str):
                            response_str = response_data
                        else:
                            response_str = str(response_data)
                        st.code(response_str, language='json')
                    else:
                        st.error("❌ No API response stored in session state. This shouldn't happen!")
                        st.info("💡 The API call might have failed before storing the response.")
                
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
