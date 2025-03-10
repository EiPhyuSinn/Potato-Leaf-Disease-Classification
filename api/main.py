import tensorflow as tf 
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import uvicorn 
from io import BytesIO
from PIL import Image 
import numpy as np

model_path = "PlantVillage/trained_model/epoch_50.keras"
model = tf.keras.models.load_model(model_path)
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'] 

app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/ping')
async def ping():
    return 'Hello there'

def preprocess(img):
    img = Image.fromarray(img).convert("RGB")  # Convert NumPy array back to PIL Image
    img = img.resize((256, 256))  # Resize to match model input
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


@app.post('/predict')
async def predict(file: UploadFile= File(...)):
    file = await file.read()
    img = np.array(Image.open(BytesIO(file)))
    img = preprocess(img)
    
    pred = model(img)
    cls = class_names[np.argmax(pred)]
    conf = np.max(pred)
    return {'Class': cls, 
            'Conf' : float(conf)}


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


