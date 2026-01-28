import os
import cv2
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
SELFIE_DIR = SRC_DIR / "data" / "selfies"
SELFIE_DIR.mkdir(parents=True, exist_ok=True)


def capture_selfie(name: str) -> str:
    """Capture a selfie using the webcam and save it to the selfies directory.
    Args:
        name (str): The name to save the selfie as (without extension).
        
    Returns:
        str: The path to the saved selfie image.        
        """
    out_path = SELFIE_DIR
    image_size = (224, 224)
    selfie_path = ""
    
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise RuntimeError("Cannot open webcam")

    print("Press 'c' to capture selfie or 'q' to quit.")
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("failed to grab frame")
            break
        
        cv2.imshow("Webcam - Press 'c' to capture", frame)
        
        key = cv2.waitKey(1) & 0xFF # Mask 
        if key == ord('c'):
            normalized_frame = cv2.resize(frame, image_size, interpolation=cv2.INTER_CUBIC)
            
            file_name = f"{name}.png"
            
            selfie_path = os.path.join(out_path, file_name)
            cv2.imwrite(selfie_path, normalized_frame)
            print(f"Selfie saved to {selfie_path}")
            break
        elif key == ord('q'):
            print("Cancelled.")
            selfie_path = None
            break
        
            
    cam.release()
    cv2.destroyAllWindows()

    return selfie_path