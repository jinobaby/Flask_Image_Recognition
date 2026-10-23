"""Load the trained Keras model and turn uploaded images into digit predictions."""
import os

from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image

# Path is relative to this file, not the author's D: drive
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "digit_model.h5")
IMAGE_SIZE = (224, 224)

model = load_model(MODEL_PATH)


def preprocess_img(img_path):
    """Open an image and convert it to a (1, 224, 224, 3) array scaled to 0-1."""
    # Convert to RGB so RGBA PNGs, greyscale and palette images all become
    # 3 channels, which is what the model's input layer expects.
    op_img = Image.open(img_path).convert("RGB")
    img_resize = op_img.resize(IMAGE_SIZE)
    img2arr = img_to_array(img_resize) / 255.0
    img_reshape = img2arr.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
    return img_reshape


def predict_result(img_array):
    """Return the most likely digit (0-9) for a preprocessed image array."""
    pred = model.predict(img_array)
    return np.argmax(pred[0], axis=-1)
