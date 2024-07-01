from fastapi import FastAPI, UploadFile, File
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from fastapi_cors import CORS
from transformers import AutoModel
from PIL import Image
import io
import torch


# Load the model
model = AutoModel.from_pretrained("model", trust_remote_code=True)
model.to("cuda")

app = FastAPI()
cors = CORS(app)

@app.get("/")
def root():
    return {
        "message": "Welcome to Radiologist",
        "description": "This API allows you to process chest X-ray images.",
        "endpoints": {
            
        }
    }
    
# Upload image and validate the file type
@app.post("/upload_image/")
async def upload_image(image: UploadFile):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, and PNG images are allowed.")
    else:
        try:
            contents = await image.read()
            cxr_image = Image.open(io.BytesIO(contents))
            return JSONResponse({"message": "Image uploaded successfully"})
        except Exception as e:
            return JSONResponse({"error": str(e)})
            
@app.post("/chatbot/")
async def chat(prompt: str, image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, and PNG images are allowed.")
    else:
        try:
            contents = await image.read()
            cxr_image = Image.open(io.BytesIO(contents))
            chat = [
            {"role": "system",
             "content": "You are a helpful radiologist. Try to interpret chest x ray image and answer to the question that user provides."},
            {"role": "user",
             "content": "<image>\n"+prompt}
        ]
            response = model.generate_cxr_repsonse(chat, cxr_image, temperature=0.2, top_p=0.8)
            return JSONResponse({"answer": response})
        except ValueError:
            return JSONResponse({"error": "No image provided, please provide an image first"})