from uuid import uuid4
from langchain_core.documents import Document
# from chunk import processed_chunks
from pdf_convertor import md_text

documents = []
# for chunk in processed_chunks:
#     document = Document(
#         page_content=chunk['text'],
#         metadata={"article_id": chunk['article_id'], "chunk_id": chunk['chunk_id'], "topic": chunk['topic']}
#     )
#     documents.append(document)

for chunk in md_text:
    document = Document(
        page_content=chunk['text'],
        metadata={"page": chunk['metadata']['page']}
    )
    documents.append(document)

uuids = [str(uuid4()) for _ in range(len(documents))]

