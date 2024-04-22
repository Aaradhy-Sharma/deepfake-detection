import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, Dense

# Custom layer for stacking
class StackLayer(Layer):
    def __init__(self, **kwargs):
        super(StackLayer, self).__init__(**kwargs)

    def call(self, inputs):
        if isinstance(inputs, list):
            return tf.stack(inputs, axis=1)
        else:
            return tf.expand_dims(inputs, axis=1)

# Step 1: Load pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
vgg16.trainable = False  # Freeze the VGG16 model

# Step 2: Define the deepfake detection model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = vgg16(inputs)
x = StackLayer()(x)
x = Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
deepfake_model = Model(inputs=inputs, outputs=outputs)

# Step 3: Detect deepfakes
def detect_deepfake(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = deepfake_model.predict(image)
    return prediction[0][0] > 0.5

# Example usage
real_count = 0
fake_count = 0
for i in range(500):
    real_path = f'Dataset/Test/Real/real_{i}.jpg'
    fake_path = f'Dataset/Test/Fake/fake_{i}.jpg'

    real_result = detect_deepfake(real_path)
    fake_result = detect_deepfake(fake_path)

    if real_result:
        print(f"\033[1;32mReal image {i}: True\033[0m")
    else:
        print(f"\033[1;32mReal image {i}: False\033[0m")
    if fake_result:
        print(f"\033[1;31mFake image {i}: True\033[0m")
    else:
        print(f"\033[1;31mFake image {i}: False\033[0m")

    if real_result:
        real_count += 1
    if fake_result:
        fake_count += 1

print(f"Correctly detected {real_count} out of 500 real images.")
print(f"Correctly detected {fake_count} out of 500 fake images.")