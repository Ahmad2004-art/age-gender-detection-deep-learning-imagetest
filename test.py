import tensorflow as tf
import numpy as np
import cv2

IMG_SIZE = (224, 224)

model = tf.keras.models.load_model("age_gender_model.keras")

img = cv2.imread("test5.jpg")
if img is None:
    raise ValueError("test.jpg not found. Put an image named test.jpg in the project folder.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, IMG_SIZE)
img = img.astype(np.float32)
img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
img = np.expand_dims(img, axis=0)

pred = model.predict(img)

gender_idx = int(np.argmax(pred["gender"][0]))
gender = "Male" if gender_idx == 1 else "Female"

age = float(pred["age"][0][0])
age_rounded = int(round(age))

print("Gender:", gender)
print("Age (raw):", age)
print("Age (rounded):", age_rounded)