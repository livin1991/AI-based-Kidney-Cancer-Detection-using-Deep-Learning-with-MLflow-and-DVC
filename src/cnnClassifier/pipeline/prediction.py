import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

        # ✅ Load model only once
        print("🔄 Loading model...")
        # self.model = load_model(os.path.join("model", "model.h5"))
        self.model = load_model("artifacts/prepare_base_model/base_model_updated.h5")
        print("✅ Model loaded")

        # ✅ Warmup (important for first prediction delay)
        dummy = np.zeros((1, 224, 224, 3))
        self.model.predict(dummy)


    def predict(self):
        print("🔹 Step 1: Loading image")

        # Load & preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)

        print("🔹 Step 2: Preprocessing")

        test_image = test_image / 255.0   # ✅ normalize (important)
        test_image = np.expand_dims(test_image, axis=0)

        print("🔹 Step 3: Predicting")

        result = np.argmax(self.model.predict(test_image), axis=1)

        print("🔹 Step 4: Done", result)

        if result[0] == 1:
            prediction = 'Tumor'
        else:
            prediction = 'Normal'

        return [{"image": prediction}]