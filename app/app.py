from fastapi import FastAPI, File, UploadFile
from PIL import Image
from wastenet.inference import InferenceSession

import io

model = InferenceSession("./models/model.onnx")

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    prediction = model.make_prediction(img)
    return {"class": prediction}
