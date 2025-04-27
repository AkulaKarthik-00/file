from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load models once
gender_model = tf.keras.models.load_model(r"C:\Users\akula\OneDrive\Desktop\GitHub\file\best_model.h5")
skin_tone_model = tf.keras.models.load_model(r"C:\Users\akula\OneDrive\Desktop\GitHub\file\skin_tone_model.h5")

@app.get("/")
async def root():
    return {"message": "Welcome to Gender and Skin Tone Detection API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((224, 224))  # Adjust if needed
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    gender_pred = gender_model.predict(img_array)
    skin_tone_pred = skin_tone_model.predict(img_array)

    gender = "Male" if np.argmax(gender_pred) == 0 else "Female"
    skin_tone_classes = ["Fair", "Medium", "Dark"]
    skin_tone = skin_tone_classes[np.argmax(skin_tone_pred)]

    return JSONResponse(content={
        "gender": gender,
        "skin_tone": skin_tone
    })
