#!/bin/bash

# Tamil Nadu Schemes Chatbot - Setup and Run Script
# This script sets up the environment and runs the bilingual FAQ chatbot

echo "🏛️ Tamil Nadu Government Schemes ChatBot Setup"
echo "தமிழ்நாடு அரசு திட்டங்கள் சாட்பாட் அமைப்பு"
echo "================================================"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements if they don't exist
echo "Installing required packages..."
pip install -q langchain langchain-community chromadb sentence-transformers openai langchain-openai python-dotenv streamlit transformers torch numpy pandas requests

# Check if datasets exist
if [ ! -d "Datasets" ]; then
    echo "❌ Error: Datasets folder not found!"
    echo "Please ensure the Datasets folder with JSON files is in the current directory."
    exit 1
fi

echo "✅ Setup complete!"
echo ""
echo "Starting the bilingual chatbot..."
echo "द्विभाषी चैटबॉट शुरू हो रहा है..."
echo ""

# Run the Streamlit app
streamlit run tn_schemes_chatbot.py --server.port 8501 --server.address localhost
