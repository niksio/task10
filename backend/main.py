from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import numpy as np
import tensorflow as tf

app = FastAPI(title="Image Classification API")

# Allow CORS for connecting frontend to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "models/best_classification_model.keras"

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

def preprocess_image(image: Image.Image):
    """
    ВНИМАНИЕ: Измените размер (`target_size`) и параметры 
    предобработки под тот датасет, с которым вы обучали модель!
    Например: (28, 28) для MNIST/FashionMNIST, (224, 224) для ImageNet.
    """
    # if image.mode != "L":
    #     image = image.convert("L")  # Перевод в градации серого. Уберите, если модель цветная (RGB)
    
    image = image.resize((224, 224))  # Размер
    
    img_array = np.array(image)
    img_array = img_array / 255.0   # Нормализация
    
    # Добавляем измерение канала (если нужно) и батча (batch, height, width, channels)
    img_array = np.expand_dims(img_array, axis=-1) 
    img_array = np.expand_dims(img_array, axis=0)
    
    # img_array.shape имеет вид (1, 224, 224, 1)
    return img_array

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=500, detail="Модель не загружена на сервере.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed_image = preprocess_image(image)
        
        # Получаем предсказания
        predictions = model.predict(processed_image)
        
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))
        all_probabilities = predictions[0].tolist()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # Запуск сервера локально для отладки
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)