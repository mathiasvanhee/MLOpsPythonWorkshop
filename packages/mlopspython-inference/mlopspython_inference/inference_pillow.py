from io import BytesIO
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Interface for loading images
class ImageLoaderInterface:
    def load_img(self, filename: str | BytesIO) -> np.ndarray:
        # Placeholder for actual image loading logic
        pass

    def img_to_array(self, img) -> np.ndarray:
        # Placeholder for actual image to array conversion
        pass

# Interface for loading images
class ImageLoaderImplementation(ImageLoaderInterface):
    def load_img(self, filename: str | BytesIO):
        img = load_img(filename, target_size=(224, 224))
        img = self.img_to_array(img)
        img = img.reshape(1, 224, 224, 3)
        # center pixel data
        img = img.astype('float32')
        img = img - [123.68, 116.779, 103.939]
        return img

    def img_to_array(self, img):
        return img_to_array(img)

# Interface to abstract the model
class ModelInterface:
    def predict(self, image_array) -> np.ndarray:
        pass

# Interface to abstract the model
class ModelImplementation(ModelInterface):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self, image_array):
        return self.model.predict(image_array)

# Inference class
class Inference:
    def __init__(self, logger, model: ModelInterface=ModelImplementation, image_loader: ImageLoaderInterface=ImageLoaderImplementation):
        self.logger = logger
        self.model = model
        self.image_loader = image_loader

    def execute(self, filepath: str | BytesIO):
        img = self.image_loader.load_img(filepath)
        result = self.model.predict(img)
        values = [float(result[0][0]), float(result[0][1]), float(result[0][2])]
        switcher = ['Cat', 'Dog', 'Other']
        prediction = np.argmax(result[0])
        return {"prediction": switcher[prediction], "values": values}

