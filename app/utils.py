import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import faiss
import pandas as pd
from PIL import Image
import numpy as np

# Cấu hình (sử dụng GPU nếu cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load FAISS index và metadata
index = faiss.read_index("heritage_image_index.faiss")
df_metadata = pd.read_csv("heritage_metadata.csv")
# Chuẩn hóa đường dẫn metadata nếu chạy trên Windows
df_metadata['file_path'] = df_metadata['file_path'].apply(os.path.normpath)

# 2. Load model và preprocess pipeline
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device).eval()
preprocess = weights.transforms()

# 3. Hàm lấy embedding từ ảnh
def get_image_embedding(image_path: str) -> np.ndarray:
    """
    Tạo embedding chuẩn hóa (float32 unit-vector) cho ảnh đầu vào.
    """
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(input_tensor).squeeze().cpu().numpy().astype('float32')
        emb /= np.linalg.norm(emb, ord=2)
    return emb

# 4. Hàm tìm kiếm ảnh tương tự
def search_image(image_path: str, top_k: int = 1, exclude_self: bool = True) -> list[str]:
    try:
        print(f"[INFO] Starting search for image: {image_path} | top_k={top_k}")
        query_path = os.path.normpath(image_path)
        q = get_image_embedding(query_path).reshape(1, -1)

        k_search = top_k + 1 if exclude_self else top_k

        distances, indices = index.search(q, k_search)
        print(f"[INFO] FAISS search completed. Found indices: {indices[0]}")

        heritage_ids = set()
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
            heritage_id = item['heritage_id']
            if heritage_id not in heritage_ids:
                heritage_ids.add(heritage_id)
            if len(heritage_ids) >= top_k:
                break

        print(f"[INFO] Search completed. Found {len(heritage_ids)} unique heritage_ids.")
        return list(heritage_ids)

    except Exception as e:
        print(f"[ERROR] search_image failed: {e}")
        raise

