# 🚀 Streamlit Cloud Deployment Guide

## 📋 Prerequisites

1. **GitHub Account** - Your code should be in a GitHub repository
2. **Streamlit Cloud Account** - Sign up at [streamlit.io/cloud](https://streamlit.io/cloud)
3. **Google OAuth Credentials** - Client ID and Secret from Google Cloud Console

## 🔧 Step-by-Step Deployment

### Step 1: Push Code to GitHub

```bash
cd /Users/shivenderabrol/Downloads/asr-testing-platform
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 2: Connect to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository: `shivender-sarvam/asr-testing-platform`
4. Set **Branch** to: `main`
5. Set **Main file path** to: `streamlit_app.py`
6. Click **"Deploy!"**

### Step 3: Configure Secrets

1. In Streamlit Cloud, go to your app's **Settings** → **Secrets**
2. Add the following secrets in **TOML format**:

```toml
GOOGLE_ID = "your-google-client-id-here"
GOOGLE_SECRET = "your-google-client-secret-here"
GOOGLE_REDIRECT_URI = "https://your-app-name.streamlit.app/"
```

**⚠️ Important:**
- Replace `your-app-name` with your actual Streamlit app name
- No brackets `[]` needed - just the key-value pairs
- Make sure there are no extra spaces or quotes around values

### Step 4: Update Google OAuth Redirect URI

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services** → **Credentials**
3. Click on your OAuth 2.0 Client ID
4. Add your Streamlit app URL to **Authorized redirect URIs**:
   ```
   https://your-app-name.streamlit.app/
   ```
5. Click **Save**

### Step 5: Test Your App

1. Wait 2-3 minutes for deployment to complete
2. Visit your app URL: `https://your-app-name.streamlit.app/`
3. Test Google OAuth login
4. Test CSV upload and ASR testing flow

## 📁 File Structure

```
asr-testing-platform/
├── streamlit_app.py          # Main app file
├── requirements_streamlit.txt # Dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README.md
```

## 🔍 Troubleshooting

### Issue: "Google OAuth not configured"
- **Solution**: Check that secrets are set correctly in Streamlit Cloud
- Make sure there are no extra brackets or formatting issues

### Issue: "Invalid redirect URI"
- **Solution**: Update Google Cloud Console with the correct Streamlit app URL
- Make sure the URL matches exactly (including trailing slash)

### Issue: App not deploying
- **Solution**: Check that `streamlit_app.py` is in the root directory
- Verify `requirements_streamlit.txt` has all dependencies

### Issue: CSV upload not working
- **Solution**: Make sure CSV has required columns: `crop_name`, `crop_code` (optional), `language` (optional)
- Column names are flexible (spaces, case-insensitive)

## 📝 CSV Format

Your CSV should have these columns (flexible naming):

```
serial_number, crop_code, crop_name, language, project
1, RICE001, Basmati Rice, hi, DCS
2, WHEAT001, Wheat, hi, DCS
```

**Note:** Column names can vary:
- `crop_name`, `Crop Name`, `crop name` all work
- `crop_code`, `Crop Code`, `crop code` all work

## 🎯 Features

✅ Google OAuth Authentication
✅ QA Name Input
✅ Language Selection
✅ CSV Upload with Flexible Column Names
✅ ASR Testing Interface
✅ Results Export (CSV Download)
✅ Sarvam Branding

## 🆘 Need Help?

If you encounter issues:
1. Check Streamlit Cloud logs (Settings → Logs)
2. Verify secrets are correctly formatted
3. Ensure Google OAuth redirect URI matches your app URL
4. Test locally first: `streamlit run streamlit_app.py`

