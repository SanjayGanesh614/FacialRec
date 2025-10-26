

# 🧠 Face Recognition Module — Project Documentation

## 📘 Overview

This module integrates **pretrained face detection and recognition models** from the open-source library [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) into your existing project.
It allows you to:

* Detect faces from images or video frames
* Generate **512-dimensional face embeddings** using pretrained models (`InceptionResnetV1`)
* Store embeddings in your database (e.g., MongoDB with vector index support)
* Perform **face matching**, **authentication**, or **clustering**

---

## 🧩 Architecture Overview

```text
          ┌───────────────────────────┐
          │  User Uploads Image/Video │
          └────────────┬──────────────┘
                       │
                       ▼
             ┌─────────────────┐
             │  MTCNN (FaceNet)│ → Detect & crop faces
             └───────┬─────────┘
                     │
                     ▼
     ┌────────────────────────────────────┐
     │ InceptionResnetV1 (Pretrained)     │ → Generate 512-d embeddings
     │ pretrained='vggface2'              │
     └────────────────┬───────────────────┘
                      │
                      ▼
       ┌──────────────────────────────┐
       │ Store embeddings in MongoDB  │
       │ + link with user IDs          │
       └──────────────────────────────┘
                      │
                      ▼
      ┌───────────────────────────────┐
      │ Compare embeddings (matching) │
      └───────────────────────────────┘
```

---

## ⚙️ 1. Setup and Installation

### Prerequisites

* Python ≥ 3.8
* PyTorch ≥ 1.8
* GPU (optional but recommended)
* Pillow, NumPy, facenet-pytorch, pymongo (if using MongoDB)

### Installation

```bash
pip install facenet-pytorch torch torchvision pillow numpy pymongo
```

or clone the repository if you want local control:

```bash
git clone https://github.com/timesler/facenet-pytorch.git
cd facenet-pytorch
pip install -e .
```

---

## 🧠 2. Core Components

### **(a) Face Detection — MTCNN**

* Detects and crops faces from an image.
* Handles resizing, margin, and alignment automatically.

### **(b) Face Embedding — InceptionResnetV1**

* Converts a face image (160×160 px) into a **512-dimensional embedding** vector.
* Pretrained on:

  * `vggface2` (best for general use)
  * `casia-webface` (optional alternative)

---

## 💻 3. Project File Structure

```bash
your_project/
│
├── face_processor/
│   ├── __init__.py
│   ├── detector.py           # Face detection using MTCNN
│   ├── embedder.py           # Embedding extraction using InceptionResnetV1
│   ├── utils.py              # Helper functions
│   └── requirements.txt
│
├── app.py                    # Example usage / integration script
└── database/
    ├── mongo_manager.py      # MongoDB + vector embedding storage
    └── __init__.py
```

---

## 🧩 4. Core Code Implementation

### **detector.py**

```python
from facenet_pytorch import MTCNN
from PIL import Image
import torch

class FaceDetector:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=0, device=self.device)

    def detect_face(self, image_path):
        img = Image.open(image_path).convert('RGB')
        face_tensor = self.mtcnn(img)
        if face_tensor is None:
            print("⚠️ No face detected.")
            return None
        return face_tensor
```

---

### **embedder.py**

```python
from facenet_pytorch import InceptionResnetV1
import torch
import numpy as np

class FaceEmbedder:
    def __init__(self, pretrained_model='vggface2', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained=pretrained_model).eval().to(self.device)

    def get_embedding(self, face_tensor):
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(face_tensor)
        return embedding.cpu().numpy()[0]

    def compare_embeddings(self, emb1, emb2, metric='cosine'):
        if metric == 'cosine':
            emb1_norm, emb2_norm = emb1 / np.linalg.norm(emb1), emb2 / np.linalg.norm(emb2)
            return np.dot(emb1_norm, emb2_norm)
        elif metric == 'euclidean':
            return np.linalg.norm(emb1 - emb2)
```

---

### **app.py — Example Integration**

```python
from face_processor.detector import FaceDetector
from face_processor.embedder import FaceEmbedder

detector = FaceDetector()
embedder = FaceEmbedder()

face1 = detector.detect_face('images/user1.jpg')
face2 = detector.detect_face('images/user2.jpg')

if face1 is not None and face2 is not None:
    emb1 = embedder.get_embedding(face1)
    emb2 = embedder.get_embedding(face2)

    similarity = embedder.compare_embeddings(emb1, emb2, metric='cosine')
    print(f"Cosine Similarity: {similarity:.4f}")

    if similarity > 0.8:
        print("✅ Match detected!")
    else:
        print("❌ No match found.")
```

---

## 🧮 5. Using AI Code Editors (e.g. GitHub Copilot, Cursor)

### 🔧 Steps to Use AI Code Assistants

1. **Open this project folder** in your IDE (VSCode / Cursor / PyCharm).

2. Add a `.copilot.json` or `.cursor.json` with instructions such as:

   ```json
   {
     "context": "This is a face recognition project using facenet-pytorch pretrained models.",
     "goals": [
       "Add routes for user registration with face capture",
       "Implement REST API for face authentication",
       "Integrate MongoDB for embedding storage and retrieval"
     ]
   }
   ```

3. Start typing function docstrings like:

   ```python
   def register_user_with_face(image_path, user_data):
       """Detect face, generate embedding, and store in MongoDB."""
   ```

   → Copilot will auto-generate full implementations using your context.

4. You can then extend it into:

   * `Flask` or `FastAPI` for backend APIs
   * `React` or `Next.js` frontend for capturing/uploading images
   * AI editors will follow this doc’s structure for consistent codegen.

---

## 🗄️ 6. Database Integration (MongoDB Example)

```python
# database/mongo_manager.py
from pymongo import MongoClient
import numpy as np

class MongoDBManager:
    def __init__(self, uri="mongodb://localhost:27017/", db_name="faceDB"):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.users = self.db["users"]

    def store_embedding(self, user_id, embedding):
        self.users.update_one(
            {"user_id": user_id},
            {"$set": {"embedding": embedding.tolist()}},
            upsert=True
        )

    def fetch_all_embeddings(self):
        return list(self.users.find({}, {"user_id": 1, "embedding": 1, "_id": 0}))
```

---

## 🧪 7. Testing the Pipeline

Run this in terminal:

```bash
python app.py
```

Expected output:

```
Cosine Similarity: 0.8734
✅ Match detected!
```

If you run with different people’s photos:

```
Cosine Similarity: 0.5123
❌ No match found.
```

---

## 🚀 8. Scaling and Optimization

| Goal            | Suggestion                                            |
| --------------- | ----------------------------------------------------- |
| Fast search     | Use **MongoDB Atlas Vector Search** or **Faiss**      |
| Real-time video | Batch frames; detect once every few frames            |
| Privacy         | Encrypt embeddings before storage                     |
| Model Serving   | Export via TorchScript or use a Flask/FastAPI wrapper |
| Edge use        | Quantize model or use lightweight CNN variants        |

---

## 🧠 9. Optional: Fine-Tuning

If you have your own dataset:

```python
resnet = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=n_classes)
# Train using your dataset (cross-entropy loss or triplet loss)
```

This allows the system to specialize in your user domain (e.g., company faces).

---

## 🧾 10. Summary

| Component           | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| **Library**         | [facenet-pytorch](https://github.com/timesler/facenet-pytorch) |
| **Detection Model** | MTCNN                                                          |
| **Embedding Model** | InceptionResnetV1 (VGGFace2 pretrained)                        |
| **Output**          | 512-dim vector per face                                        |
| **Integration**     | Works with MongoDB, REST APIs, and authentication pipelines    |
| **AI Code Editors** | Can auto-generate additional features from this structured doc |

---

