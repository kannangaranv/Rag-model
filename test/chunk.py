import pandas as pd

# Load data from CSV file
df = pd.read_csv('C:\\Users\\nuwank\\Documents\\BoardPAC\\Development\\Rag-model\\dataset\\rag_sample_qas_from_kis.csv')

articles = df['ki_text'].tolist()
topics = df['ki_topic'].tolist()
questions = df.iloc[:, 2].tolist()  
ground_truths = df.iloc[:, 3].tolist()  

# print(f" Loaded {len(articles)} articles")
# print(f"Columns: {df.columns.tolist()}")

def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:
            chunks.append(chunk)
    return chunks

# Process articles into chunks
processed_chunks = []
for idx, article in enumerate(articles):
    chunks = chunk_text(article)
    for chunk_idx, chunk in enumerate(chunks):
        processed_chunks.append({
            'article_id': idx,
            'chunk_id': chunk_idx,
            'text': chunk,
            'topic': topics[idx]
        })

# chunk_texts = [chunk['text'] for chunk in processed_chunks]
# print(f" Created {len(chunk_texts)} chunks")
# print (f"processed_chunks: {processed_chunks[:5]}")  