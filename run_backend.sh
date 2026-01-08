#!/bin/bash
# Run StratAI Backend

echo "🚀 Starting StratAI Backend..."

# Activate venv
source venv/bin/activate

# Run backend
python -m uvicorn backend.main:app --reload --port 8000
