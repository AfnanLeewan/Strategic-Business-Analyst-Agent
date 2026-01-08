#!/bin/bash
# Run StratAI Frontend

echo "🎨 Starting StratAI Frontend..."

# Activate venv
source venv/bin/activate

# Run frontend
streamlit run frontend/app.py --server.port 8501
