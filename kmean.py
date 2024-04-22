import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

NUM_TESTS = 1000

# Load pre-trained VGG16 model
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Initialize features and labels for initialization
init_features = []
init_labels = []

# Add 10 real and 10 fake images for initialization
for i in range(100):
    real_path = f'Dataset/Test/Real/real_{i}.jpg'
    fake_path = f'Dataset/Test/Fake/fake_{i}.jpg'

    real_image = cv2.imread(real_path)
    real_image = cv2.resize(real_image, (224, 224))
    real_image = np.expand_dims(real_image, axis=0)
    real_feature = vgg16.predict(real_image)
    init_features.append(real_feature.flatten())
    init_labels.append(0)  # 0 for real

    fake_image = cv2.imread(fake_path)
    fake_image = cv2.resize(fake_image, (224, 224))
    fake_image = np.expand_dims(fake_image, axis=0)
    fake_feature = vgg16.predict(fake_image)
    init_features.append(fake_feature.flatten())
    init_labels.append(1)  # 1 - fake 

# Calculating initial cluster centers
init_features = np.stack(init_features)
real_init_center = np.mean(init_features[np.array(init_labels) == 0], axis=0)
fake_init_center = np.mean(init_features[np.array(init_labels) == 1], axis=0)
init_centers = np.stack([real_init_center, fake_init_center])

# Initialize k-means with the initial centers
kmeans = KMeans(n_clusters=2, init=init_centers, n_init=1, random_state=42)
kmeans.fit(init_features)

# Plot the initial clusters
real_cluster = init_features[np.array(init_labels) == 0]
fake_cluster = init_features[np.array(init_labels) == 1]

plt.scatter(real_cluster[:, 0], real_cluster[:, 1], c='r', label='Real')
plt.scatter(fake_cluster[:, 0], fake_cluster[:, 1], c='b', label='Fake')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='k', marker='x', s=100, label='Centroids')
plt.legend()
plt.title('Initial Clusters')
plt.show()

# Classify remaining images based on distance from centroids
features = []
labels = []
for i in range(100, NUM_TESTS):
    real_path = f'Dataset/Test/Real/real_{i}.jpg'
    fake_path = f'Dataset/Test/Fake/fake_{i}.jpg'

    real_image = cv2.imread(real_path)
    real_image = cv2.resize(real_image, (224, 224))
    real_image = np.expand_dims(real_image, axis=0)
    real_feature = vgg16.predict(real_image)
    features.append(real_feature.flatten())
    labels.append(0)  # 0 for real

    fake_image = cv2.imread(fake_path)
    fake_image = cv2.resize(fake_image, (224, 224))
    fake_image = np.expand_dims(fake_image, axis=0)
    fake_feature = vgg16.predict(fake_image)
    features.append(fake_feature.flatten())
    labels.append(1)  # 1 for fake

features = np.stack(features)
labels = np.array(labels)

# Classify based on distance from centroids
predictions = kmeans.predict(features)

# Evaluate performance
real_correct = np.sum((predictions == 0) & (labels == 0))
fake_correct = np.sum((predictions == 1) & (labels == 1))

print(f"Correctly classified {real_correct} out of {NUM_TESTS - 10} real images.")
print(f"Correctly classified {fake_correct} out of {NUM_TESTS - 10} fake images.")

# Plot the final clusters
real_cluster = features[np.array(labels) == 0]
fake_cluster = features[np.array(labels) == 1]

plt.figure()
plt.scatter(real_cluster[:, 0], real_cluster[:, 1], c='r', label='Real')
plt.scatter(fake_cluster[:, 0], fake_cluster[:, 1], c='b', label='Fake')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='k', marker='x', s=100, label='Centroids')
plt.legend()
plt.title('Final Clusters')
plt.show()