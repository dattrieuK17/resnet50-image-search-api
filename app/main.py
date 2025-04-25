from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from app.utils import search_image

app = FastAPI()
@app.get("/")
def root():
    return {"message": "Server is running."}

@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    # Lưu ảnh tạm thời
    temp_path = f"/tmp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Tìm kiếm ảnh tương tự
    try:
        results_df = search_image(temp_path, top_k=top_k)
        os.remove(temp_path)
        return results_df.to_dict(orient="records")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# uvicorn app.main:app --host 0.0.0.0 --port 8000
# Truy cập API tại: http://localhost:8000/docs để test giao diện Swagger UI.
