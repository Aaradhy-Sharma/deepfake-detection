import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Step 3: Compile the model
deepfake_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 4: Prepare the data
train_dir = 'Dataset/Train'
test_dir = 'Dataset/Test'
val_dir = 'Dataset/Validation'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Step 5: Train the model
deepfake_model.fit(
    train_generator,
    epochs=10,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator))

# Step 6: Evaluate the model
loss, accuracy = deepfake_model.evaluate(test_generator, steps=len(test_generator))
print(f"Test loss: {loss:.2f}, Test accuracy: {accuracy:.2f}")

# Step 7: Detection of deepfakes
def detect_deepfake(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = deepfake_model.predict(image)
    return prediction[0][0] > 0.5

# Example usage
print(detect_deepfake('Dataset/Test/Real/real_0.jpg'))  # False (real image)
print(detect_deepfake('Dataset/Test/Fake/fake_0.jpg'))  # True (deepfake image)