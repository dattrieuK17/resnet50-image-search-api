import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import faiss
import pandas as pd
from PIL import Image
import numpy as np

# Khởi tạo biến toàn cục rỗng
model = None
preprocess = None
index = None
df_metadata = None

def load_model():
    global model, preprocess
    if model is None:
        print("[INFO] Loading ResNet50 model and preprocessing transforms...")
        weights = ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model = torch.nn.Sequential(*(list(model.children())[:-1])).eval()
        preprocess = weights.transforms()
        print("[INFO] Model and transforms loaded successfully.")
    else:
        print("[INFO] Model already loaded.")

def load_faiss_index():
    global index, df_metadata
    if index is None or df_metadata is None:
        print("[INFO] Loading FAISS index and metadata CSV...")
        try:
            index = faiss.read_index("heritage_image_index.faiss")
            df_metadata = pd.read_csv("heritage_metadata.csv")
            df_metadata['file_path'] = df_metadata['file_path'].apply(os.path.normpath)
            print(f"[INFO] Loaded FAISS index and metadata with {len(df_metadata)} entries.")
        except Exception as e:
            print("[ERROR] Failed to load FAISS index or metadata:", e)
            raise

def get_image_embedding(image_path: str) -> np.ndarray:
    try:
        load_model()
        print(f"[INFO] Loading and preprocessing image: {image_path}")
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            emb = model(input_tensor).squeeze().numpy().astype('float32')
            emb /= np.linalg.norm(emb, ord=2)
        print(f"[INFO] Image embedding created for: {image_path}")
        return emb
    except Exception as e:
        print(f"[ERROR] Error generating embedding for {image_path}: {e}")
        raise

def search_image(image_path: str, top_k: int = 5, exclude_self: bool = True) -> pd.DataFrame:
    try:
        print(f"[INFO] Starting search for image: {image_path} | top_k={top_k}")
        load_faiss_index()
        query_path = os.path.normpath(image_path)
        q = get_image_embedding(query_path).reshape(1, -1)

        k_search = top_k + 1 if exclude_self else top_k
        distances, indices = index.search(q, k_search)
        print(f"[INFO] FAISS search completed. Found indices: {indices[0]}")

        results = []
        self_idx = None
        if exclude_self:
            mask = df_metadata['file_path'] == query_path
            if mask.any():
                self_idx = df_metadata.index[mask][0]
                print(f"[INFO] Self index detected: {self_idx}")

        for dist, idx in zip(distances[0], indices[0]):
            if idx == self_idx:
                continue
            item = df_metadata.iloc[idx]
            results.append({
                'heritage_name': item['heritage_name'],
                'file_path': item['file_path'],
                'similarity': float(dist)
            })
            if len(results) >= top_k:
                break

        print(f"[INFO] Search completed. Returning {len(results)} results.")
        return pd.DataFrame(results)

    except Exception as e:
        print(f"[ERROR] search_image failed: {e}")
        raise
