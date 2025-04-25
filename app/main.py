from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from app.utils import search_image
import traceback

app = FastAPI()

@app.get("/")
def root():
    print("[INFO] Root endpoint called.")
    return {"message": "Server is running."}

@app.post("/search")
async def search(file: UploadFile = File(...), top_k: int = 5):
    print(f"[INFO] /search endpoint called with file: {file.filename}, top_k: {top_k}")
    
    temp_path = f"/tmp/{file.filename}"
    os.makedirs("/tmp", exist_ok=True)

    try:
        # Ghi ảnh vào thư mục tạm
        print(f"[INFO] Saving uploaded file to: {temp_path}")
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print("[INFO] File saved successfully.")

        # Thực hiện tìm kiếm
        print("[INFO] Running search_image...")
        results_df = search_image(temp_path, top_k=top_k)
        print("[INFO] Search completed.")

        # Xóa ảnh tạm
        os.remove(temp_path)
        print("[INFO] Temporary file removed.")

        return results_df.to_dict(orient="records")

    except Exception as e:
        print("[ERROR] Exception occurred in /search endpoint:")
        traceback.print_exc()
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("[INFO] Temporary file removed after error.")
        return JSONResponse(status_code=500, content={"error": str(e)})
