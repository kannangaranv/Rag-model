# RAG Model  
RAG Model Development Guide

## Prerequisites  
- Python 3.8 or above  
- Installed `pip`  
- `.env` file configured with necessary environment variables  

---

## 1. Import the `.env` file  
Make sure a `.env` file exists in the root directory with all required environment variables.  

---

## 2. Install Packages  
Run the following command to install all dependencies:  
```bash
pip install -r requirements.txt
```

## 3. Upload Embeddings to Vector Store
```bash
python upload.py
```

## 4. Start the RAG model
```bash
python main.py
```

Install & Link FFmpeg Correctly
🔧 1. Download FFmpeg (Windows)

Go to: https://www.gyan.dev/ffmpeg/builds/

Download the Release full build ZIP

Extract the ZIP to:
C:\ffmpeg

2. Add FFmpeg to Your System PATH

Press Win + S, search for Environment Variables

Click "Edit the system environment variables"

Click Environment Variables

Under System Variables, find and edit the Path

Add this new entry:

C:\ffmpeg\bin