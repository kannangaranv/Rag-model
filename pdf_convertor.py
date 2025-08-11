import pymupdf4llm
import json



md_text = pymupdf4llm.to_markdown(
    "C:\\Users\\nuwank\\Documents\\BoardPAC\\Development\\Rag-model\\dataset\\BoardPAC_User Manual_Secretary_V4.2.10000.pdf", 
    page_chunks=True, 
    # write_images=True,
    # image_path="C:\\Users\\nuwank\\Documents\\BoardPAC\\Development\\Rag-model\\dataset\\images",
    # image_format="png"
)

# simplified_chunks = []

# for chunk in md_text_images:
#     simplified_chunks.append({
#         "page": chunk["metadata"]["page"],
#         "text": chunk["text"]
#     })

# with open("all_chunks_simplified.json", "w", encoding="utf-8") as f:
#     json.dump(simplified_chunks, f, ensure_ascii=False, indent=2)
