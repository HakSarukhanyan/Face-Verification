
from retinaface import RetinaFace
import cv2
from pathlib import Path


def find_face(image_path: str) -> dict:
    """
    Detect a face in the image and return its details.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        dict: A dictionary containing facial area, landmarks, and confidence score.
    """
    faces = RetinaFace.detect_faces(image_path)

    if len(faces) != 1:
        raise ValueError("Image must contain exactly one face")

    face_data = next(iter(faces.values()))

    return {
        "facial_area": face_data["facial_area"],
        "landmarks": face_data["landmarks"],
        "confidence": face_data["score"]
    }


def crop_face(image_path: str, face_info: dict) -> Path:
    """
    Crop and normalize the face from the image based on detected face information.
    
    Args:
        image_path (str): Path to the image file.
        face_info (dict): Dictionary containing facial area and landmarks.
    
    Returns:
        Path: The path to the cropped face image.
    """
    SRC_DIR = Path(__file__).resolve().parents[1]
    OUT_DIR = SRC_DIR / "data" / "cropped_faces"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_size = (224, 224)
    padding = 0.25

    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Image not found or unreadable")

    img_h, img_w = image.shape[:2]

    x, y, w, h = face_info["facial_area"]

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    face = image[y1:y2, x1:x2]
    normalized_face = cv2.resize(
        face, image_size, interpolation=cv2.INTER_CUBIC
    )

    output_path = OUT_DIR / f"face_{image_path.name}"
    cv2.imwrite(str(output_path), normalized_face)

    return output_path
