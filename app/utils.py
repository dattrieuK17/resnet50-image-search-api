import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
import faiss
import pandas as pd
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import tqdm
import uuid


def crawl_data(heritage_name_f, output_dir, num_images):
    # Danh sách các địa điểm di tích lịch sử
    heritage_list = []

    with open(heritage_name_f, "r", encoding='utf-8') as f:
        for heritage_name in f:
            heritage_list.append(heritage_name.strip())

    heritage_list.sort()

    for heritage in heritage_list:
        # Đổi tên thư mục con cho từng di tích
        crawler = GoogleImageCrawler(storage={'root_dir': f'{output_dir}/{heritage}'})
        # crawler = BingImageCrawler(storage={'root_dir': f'{output_dir}/{"Chợ Bến Thành"}'}) # Có thể thay thành Bing nếu xảy ra lỗi
        crawler.crawl(keyword=heritage, max_num=num_images)

def create_database():
    # Cấu hình (sử dụng GPU nếu cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Ánh xạ name -> id
    heritage_names = os.listdir('heritage_images')
    heritage2id = {name: idx for idx, name in enumerate(heritage_names)}

    # 2. Chuẩn bị model và preprocess
    weight = ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weight)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Loại bỏ lớp FCFC
    model.to(device).eval()

    # Pipeline preprocess chuẩn của ResNet50
    preprocess = weight.transforms()

    # 3. Tạo embedding vector từ ảnh
    embeddings = []
    metadata = []

    for heritage_name in tqdm(heritage_names):
        heritage_id = heritage2id[heritage_name]
        folder = os.path.join("heritage_images", heritage_name)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                image = Image.open(fpath).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model(input_tensor)                  # shape [1, 2048, 1, 1]
                    emb = emb.squeeze()                         # shape [2048]
                    emb = emb.cpu().numpy().astype('float32')   # đảm bảo float32 dtype
                    emb /= np.linalg.norm(emb, ord=2)           # chuẩn hóa về vector đơn vị
            except Exception as e:
                print(f"Error processing {fpath}: {e}")
                continue

            img_id = str(uuid.uuid4())
            embeddings.append(emb)
            metadata.append({
                'image_id': img_id,
                'heritage_id': heritage_id,
                'heritage_name': heritage_name,
                'file_path':       os.path.normpath(fpath)
            })

    # 4. Tạo FAISS index
    embedding_dim = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(embedding_dim)

    # Thêm embeddings vào index
    embedding_matrix = np.vstack(embeddings)  # shape [N, D]
    index.add(embedding_matrix)

    # 5. Lưu index và metadata
    faiss.write_index(index, "heritage_image_index.faiss")

    df_metadata = pd.DataFrame(metadata)
    df_metadata.to_csv("heritage_metadata.csv", index=False)

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

def search_image(image_path: str, top_k: int = 5, exclude_self: bool = True) -> pd.DataFrame:
    """
    Tìm top_k ảnh tương tự từ index, loại bỏ ảnh truy vấn nếu exclude_self=True.

    Trả về DataFrame gồm: heritage_name, file_path, distance.
    """
    # Chuẩn hóa đường dẫn
    query_path = os.path.normpath(image_path)
    # Sinh vector truy vấn
    q = get_image_embedding(query_path).reshape(1, -1)
    # Search top_k (+1 để loại self)
    k_search = top_k + 1 if exclude_self else top_k
    distances, indices = index.search(q.astype('float32'), k_search)

    results = []
    # Tìm i_self để loại bỏ
    self_idx = None
    if exclude_self:
        mask = df_metadata['file_path'] == query_path
        if mask.any():
            self_idx = df_metadata.index[mask][0]
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx == self_idx:
            continue
        item = df_metadata.iloc[idx]
        results.append({
            'heritage_name': item['heritage_name'],
            'file_path':      item['file_path'],
            'similarity':       float(dist)
        })
        if len(results) >= top_k:
            break

    return pd.DataFrame(results)