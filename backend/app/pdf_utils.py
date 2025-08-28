import pymupdf4llm

# Convert PDF to Markdown
def convert_pdf_to_markdown(pdf_path):
    md_text = pymupdf4llm.to_markdown(pdf_path)
    return md_text