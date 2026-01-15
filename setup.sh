#!/bin/bash

# Create dependencies folder
mkdir -p external

# Download emojilib JSON
echo "📥 Downloading emojilib..."
curl -L https://raw.githubusercontent.com/muan/emojilib/refs/heads/main/dist/emoji-en-US.json -o external/emojis.json

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete."