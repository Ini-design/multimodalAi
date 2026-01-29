# multimodalAi
A multimodal Ai that can generate image, convert audio to text(english),  translate language to French and Germany and a chatbot too


metadata
title: multimodalAi 
emoji: 🐨
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
🎧 Multimodal AI Assistant
A powerful multi-modal AI assistant that can:

Convert voice notes to text
Translate text from English → French / German
Generate images from text prompts
Act as a chatbot powered by GROQ
Built with Python, Gradio, Torch, Transformers, and Hugging Face Spaces.

🧰 Features
Voice to Text
Upload voice notes and automatically transcribe them into text using Whisper.

Translation
Translate English text into French or German using Hugging Face MarianMT models.

Image Generation
Generate images from prompts using Stable Diffusion.

Chatbot
Chat with a conversational AI powered by GROQ API.

🚀 Tech Stack
Frontend / UI: Gradio
Speech-to-Text: OpenAI Whisper via Hugging Face Transformers
Image Generation: Stable Diffusion (Diffusers + Torch)
Translation: Hugging Face MarianMT
Chatbot: GROQ API
Backend: Python + Torch
📁 Project Structure
python app.py

GROQ_API_KEY=your_api_key_here
GOOGLE_API_KEY = your_api_key_here

⚡ Installation
Clone the repository:
git clone <your-repo-url>
cd multimodal-ai-assistant
Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
