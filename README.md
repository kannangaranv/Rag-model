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