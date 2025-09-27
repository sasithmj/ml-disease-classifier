from fastapi import FastAPI, File, UploadFile
import uvicorn
import joblib
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load your trained model
model = joblib.load("best_paddy_dt_model.pkl")

# Preprocessing (adjust this to how your model expects input)
def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))  # resize to model input size
    image = np.array(image) / 255.0   # normalize
    image = image.flatten().reshape(1, -1)  # flatten for ML models
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)[0]

    return {"prediction": str(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
