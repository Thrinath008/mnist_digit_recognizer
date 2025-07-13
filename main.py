from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # You can restrict to ["http://127.0.0.1:5500"] later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = load_model("digit_recognizer_model.keras")

@app.post("/predict")
async def predict_digit(image: UploadFile = File(...)):
    # Read and preprocess image
    image_bytes = await image.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28)

    # Predict digit
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    return JSONResponse(content={"predicted_digit": predicted_digit})