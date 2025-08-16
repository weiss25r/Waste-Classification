from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from wastenet.inference import InferenceSession

import io

model = InferenceSession("./models/model_large.onnx")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permette a tutte le origini
    allow_credentials=True,
    allow_methods=["*"],  # Permette tutti i metodi (POST, GET, etc.)
    allow_headers=["*"],  # Permette tutti gli header
)

@app.post("/predict/")
async def predict(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    prediction = model.make_prediction(img)
    return {"class": prediction}
