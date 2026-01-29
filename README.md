#  Face Verification via Web Camera and ID Card

A **desktop, real-time face verification system** that checks whether a person matches the face shown on their **ID card**, using computer vision and deep learning.


---

##  Project Overview

The system performs **multi-step face verification**:

1. Extracts a face from a **scanned ID card (PDF / image)**
2. Captures a **live selfie** using a webcam
3. Detects faces in all inputs
4. Generates **face embeddings**
5. Verifies whether faces belong to the **same person**

---

##  Verification Strategy

Check performed:

| Comparison | Purpose |
|----------|---------|
| **ID Face ↔ Selfie Face** | Identity verification |


✅ Final result is **MATCH** only if **check pass**

---

##  Core Technologies

- **OpenCV** – webcam & image processing
- **RetinaFace** – face detection
- **InsightFace** – face embeddings
- **Cosine Similarity** – face comparison

GPU acceleration is supported and recommended.

---

##  Repository Structure

```text
Face-Verification/
├── src/
│   ├── data/
│   │    ├──cropped_faces/
│   │    ├──images/
│   │    ├──PDFs/
│   │    ├──selfies
│   ├──UTILS/
│   │    ├── __init.py__
│   │    ├── detect_face_on_image.py  # Face detection 
│   │    ├── embedding.py           # Face embeddings     │   │    ├── pdf_to_image.py        # Convert to image
│   │    ├── image_from_webcam.py   # Webcam capture
│   └── main.py                # MAIN
├── requirments.txt
└── README.md
```

---

##  Installation

### 1️ Clone repository

git clone https://github.com/HakSarukhanyan/Face-Verification.git  
cd Face-Verification

### 2️ Create virtual environment

python -m venv venv

### 3️ Activate environment

**Windows**

venv\Scripts\activate

### 4️ Install dependencies

pip install -r requirments.txt

 Make sure your webcam is available and not used by another application.

---

##  How to Run

python src/main.py

### Runtime Flow
1. Load scanned ID card
2. Detect and extract ID face
3. Capture live selfie
4. Detect faces in each step
5. Generate embeddings
6. Verify identity

---

##  Face Detection Rules

| Input | Expected Faces |
|-----|----------------|
| ID Card | 1 face |
| Selfie | 1 face |

If face count conditions fail → verification stops.

---

##  Similarity Threshold

Cosine similarity threshold:

similarity > 0.5 → MATCH

 With **InsightFace + GPU**, values between **0.5–0.6** provide stable results.

---

##  Example Output

ID ↔ Selfie similarity: 0.72 ✅  

Final Result: MATCH

or

Final Result: NO MATCH


---

##  Limitations

- Desktop only
- Single person only
- No liveness detection (yet)
- Requires clear lighting and frontal faces

---

##  Future Improvements

- Liveness detection (blink / motion)
- Anti-spoofing
- GUI interface
- Multiple ID types

---

##  Author

**HakSarukhanyan**  
Computer Vision & Machine Learning Enthusiast
