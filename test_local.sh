#!/bin/bash
# Local testing script for ASR Testing Platform

echo "🚀 Starting ASR Testing Platform on localhost..."
echo ""
echo "📋 Prerequisites:"
echo "   1. Make sure you have your API key in .streamlit/secrets.toml"
echo "   2. Make sure Google OAuth is configured for http://localhost:8501/"
echo ""
echo "🌐 The app will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
source venv/bin/activate
streamlit run streamlit_app.py --server.port 8501 --server.address localhost

