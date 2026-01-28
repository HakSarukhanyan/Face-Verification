from insightface.app import FaceAnalysis
import cv2
import numpy as np
from UTILS.detect_face_on_image import find_face

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(224, 224)) # Use GPU with ID 0

def get_embedding(image_path: str) -> tuple[np.ndarray, float]:
    """
    Get the face embedding for a given image.
    
    Args:
    
        image_path (str): Path to the image file.
        
    Returns:
        embedding (np.ndarray): 512-d face embedding vector.
        confidence (float): Confidence score of the detected face.
    """
    
    face_info = find_face(image_path)
    x1, y1, x2, y2 = map(int, face_info["facial_area"])

    img = cv2.imread(image_path)
    face_crop = img[y1:y2, x1:x2]

    faces = app.get(face_crop)
    if len(faces) != 1:
        raise ValueError(f"No face found in cropped image {image_path}")
    
    embedding = faces[0].embedding  # normalized 512-d vector
    
    return embedding, face_info["confidence"]



def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Cosine similarity between the two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



def verify_faces(id_path: str, selfie_path: str, threshold : float = 0.52) -> tuple[bool, float]:
    """
    Verify if the face in the ID image matches the face in the selfie image.
    
    Args:
        id_path (str): Path to the ID image file.
        selfie_path (str): Path to the selfie image file.
        threshold (float): Similarity threshold for verification.
    
    Returns:
        verified (bool): True if faces match, False otherwise.
        similarity (float): Cosine similarity score between the two face embeddings.
    
    """
    emb_id, conf_id = get_embedding(id_path)
    emb_selfie, conf_selfie = get_embedding(selfie_path)

    sim_id_selfie = cosine_sim(emb_id, emb_selfie)

    verified = sim_id_selfie > threshold 

    print(f"ID ↔ Selfie similarity: {sim_id_selfie:.3f}")
    print(f"Threshold: {threshold}")
    print(f"Verification Result:{'VERIFIED' if verified else 'REJECTED'}")

    return verified, sim_id_selfie
