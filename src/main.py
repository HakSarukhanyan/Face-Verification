import os
import cv2 
import pymupdf
import numpy as np
from insightface.app import FaceAnalysis
from retinaface import RetinaFace
from pathlib import Path
from UTILS.detect_face_on_image import find_face, crop_face
from UTILS.pdf_to_image import image_from_pdf, pdf_folder, image_folder
from UTILS.embedding import get_embedding, cosine_sim, verify_faces
from UTILS.image_from_webcam import capture_selfie


print(np.__version__)
print(cv2.__version__)
print(pymupdf.__version__)
print(RetinaFace.__version__)

# BASE_DIR = Path(__file__).resolve().parent

# PDF_DIR = BASE_DIR / pdf_folder
# IMAGE_DIR = BASE_DIR / image_folder

# pdf_name = input("Enter the PDF file name (without .pdf extension): ") + ".pdf"
# pdf_path = PDF_DIR / pdf_name

# if not pdf_path.exists():
#     raise FileNotFoundError(f"{pdf_name} does not exist in {PDF_DIR}.")

# print(f"Using PDF: {pdf_path}")


# image_name = image_from_pdf(str(pdf_path))
# image_path = IMAGE_DIR / image_name

# print(f"Converted PDF to image: {image_path}")


# face_data = find_face(str(image_path))
# new_image = crop_face(str(image_path), face_data)


# selfie = capture_selfie(image_path.stem + "_selfie")

# verify_faces(str(image_path), str(selfie), threshold=0.5)