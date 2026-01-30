# import pymupdf4llm

# # Convert PDF to Markdown
# def convert_pdf_to_markdown(pdf_path):
#     md_text = pymupdf4llm.to_markdown(pdf_path)
#     return md_text

from pathlib import Path
from io import BytesIO
from typing import Union, BinaryIO

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Initialize once to avoid reloading models each call
_converter = PdfConverter(artifact_dict=create_model_dict())

def convert_pdf_to_markdown(pdf: Union[str, Path, bytes, BinaryIO]) -> str:
    
    if isinstance(pdf, Path):
        pdf = str(pdf)               
    elif isinstance(pdf, (bytes, bytearray)):
        pdf = BytesIO(pdf)           # wrap raw bytes

    rendered = _converter(pdf)       # str path or BytesIO/file-like
    md_text, _, _ = text_from_rendered(rendered)
    return md_text
