#!/bin/bash

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete."
