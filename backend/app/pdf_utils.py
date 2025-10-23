# import pymupdf4llm

# # Convert PDF to Markdown
# def convert_pdf_to_markdown(pdf_path):
#     md_text = pymupdf4llm.to_markdown(pdf_path)
#     return md_text

# pip install marker-pdf  (requires Python 3.10+ and PyTorch)
from pathlib import Path
from io import BytesIO
from typing import Union, BinaryIO

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

# Initialize once to avoid reloading models each call
_converter = PdfConverter(artifact_dict=create_model_dict())

def convert_pdf_to_markdown(pdf: Union[str, Path, bytes, BinaryIO]) -> str:
    """
    Accepts: str path, Path, raw bytes, or a file-like (BytesIO).
    Returns: Markdown text extracted by Marker.
    """
    if isinstance(pdf, Path):
        pdf = str(pdf)               # <- fix: Marker expects str or BytesIO
    elif isinstance(pdf, (bytes, bytearray)):
        pdf = BytesIO(pdf)           # wrap raw bytes

    rendered = _converter(pdf)       # str path or BytesIO/file-like
    md_text, _, _ = text_from_rendered(rendered)
    return md_text
