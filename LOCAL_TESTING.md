# 🧪 Local Testing Guide

## Quick Start

1. **Add your API key to secrets:**
   ```bash
   # Edit .streamlit/secrets.toml and add your SARVAM_API_KEY
   nano .streamlit/secrets.toml
   ```

2. **Run the test script:**
   ```bash
   ./test_local.sh
   ```

   Or manually:
   ```bash
   source venv/bin/activate
   streamlit run streamlit_app.py
   ```

3. **Open in browser:**
   - The app will automatically open at `http://localhost:8501`
   - Or manually navigate to that URL

## What to Test

1. ✅ **Login** - Google OAuth should work (make sure redirect URI is set to `http://localhost:8501/` in Google Console)
2. ✅ **CSV Upload** - Upload a test CSV with crop names
3. ✅ **Audio Recording** - Record audio and click "Submit Recording"
4. ✅ **ASR Processing** - Should show transcript and Yes/No immediately
5. ✅ **Navigation** - Click "Continue to Attempt X" or "Next Crop" buttons
6. ✅ **CSV Download** - Click "End Session & Get Reports" and download CSV

## Debugging

- Check the "🔍 Debug Info" expander to see what's happening
- Check browser console (F12) for JavaScript errors
- Check terminal for Python errors

## Common Issues

- **Audio not processing?** Check if API key is correct in `secrets.toml`
- **Buttons not working?** Check browser console for JavaScript errors
- **OAuth not working?** Make sure redirect URI in Google Console matches `http://localhost:8501/`

