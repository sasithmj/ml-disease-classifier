from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import joblib
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Load your trained model safely
try:
    model = joblib.load("best_paddy_rf_model.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None

def preprocess_image(image: Image.Image):
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = image.flatten().reshape(1, -1)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Model not loaded."}
        )
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        processed = preprocess_image(image)
        prediction = model.predict(processed)[0]
        return {"prediction": str(prediction)}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
