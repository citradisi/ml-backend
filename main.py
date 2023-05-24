import uvicorn
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Path, Query
import tensorflow as tf
import numpy as np
import os
from keras.models import load_model
# import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

app = FastAPI()  # create a new FastAPI app instance

# Define a Pydantic model for an item

port = int(os.getenv("PORT"))


class Item(BaseModel):
    scan_food_image: str

# model = tf.keras.models.load_model('./model/somethingidk.h5')


def predict(scan_food_image):
    category = [
        'asinan-jakarta',
        'ayam-betutu',
        'bika-ambon',
        'bubur-manado',
        'es-dawet',
        'gado-gado',
        'gudeg',
        'gulai-ikan-mas',
        'kerak-telor',
        'mie-aceh',
        'nasi-goreng-kampung',
        'rawon',
        'rendang',
        'sate',
        'soto',
    ]

    image_path = 'https://storage.googleapis.com/ariamaulana/' + scan_food_image
    response = requests.get(image_path)
    # model = load_model('/content/drive/MyDrive/Model/Models.h5')
    model = load_model('./model/Model.h5')

    img = Image.open(BytesIO(response.content))
    # img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize((150, 150))
    img = np.array(img) / 255
    img = np.expand_dims(img, axis=0)

    predict = model.predict(img)
    predict = np.argmax(predict)
    predict = category[predict]
    return predict

@app.get("/")
def hello_world():
    return ("hello world")


@app.post("/")
def add_item(item: Item):
    result = predict(item.scan_food_image)
    return {result}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)
