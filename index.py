from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
from pathlib import Path

app = FastAPI()

model = load_model("face_antispoofing_model.h5")

IMG_WIDTH = 224
IMG_HEIGHT = 224
THRESHOLD = 0.5
TEMP_DIR = Path("temp")

TEMP_DIR.mkdir(exist_ok=True)

@app.post("/verify")
async def verify_image(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        file_extension = file.filename.split(".")[-1]
        temp_file_name = f"{uuid.uuid4()}.{file_extension}"
        temp_file_path = TEMP_DIR / temp_file_name

        with temp_file_path.open("wb") as buffer:
            buffer.write(await file.read())

        image = load_img(temp_file_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        image_array = img_to_array(image)
        image_array = image_array / 255.0  # Normalize
        image_array = np.expand_dims(image_array, axis=0)
        
        prediction = model.predict(image_array)[0][0]
        
        is_real = bool(prediction <= THRESHOLD)  
        result = "Verified" if is_real else "Not Verified"
        
        response_data = {
            "filename": file.filename,
            "prediction_score": float(prediction),
            "is_real": is_real,
            "result": result
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

    finally:
        if temp_file_path and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error deleting temporary file: {str(e)}")

@app.on_event("startup")
async def startup_event():
    print(f"Temporary directory is set to: {TEMP_DIR}")
    print("Ensuring temporary directory is empty...")
    for file in TEMP_DIR.glob("*"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error removing file {file}: {str(e)}")
    print("Temporary directory is empty and ready for use.")

@app.on_event("shutdown")
async def shutdown_event():
    print("Cleaning up temporary directory...")
    for file in TEMP_DIR.glob("*"):
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error removing file {file}: {str(e)}")
    print("Temporary directory cleaned.")