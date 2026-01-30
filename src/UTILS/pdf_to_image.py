from pathlib import Path
import pymupdf
import os

image_folder = "data/images/"
pdf_folder = "data/PDFs/"

def image_from_pdf(pdf_path: str) -> str:
    """Convert the first page of a PDF to an image and save it.
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Name of the saved image file.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"{pdf_path} does not exist")

    output_dir = Path(__file__).resolve().parents[1] / image_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_document = pymupdf.open(str(pdf_path))
    page = pdf_document.load_page(0)
    pix = page.get_pixmap()

    image_name = pdf_path.stem + ".png"
    image_path = output_dir / image_name
    
    pix.save(str(image_path))
    return image_name
