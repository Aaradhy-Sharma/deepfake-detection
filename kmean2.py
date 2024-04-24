import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

NUM_TESTS = 1000

# Load pre-trained models
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Initialize combined features
features = []

# Extract combined features from all images
for i in range(NUM_TESTS):
    real_path = f'Dataset/Test/Real/real_{i}.jpg'
    fake_path = f'Dataset/Test/Fake/fake_{i}.jpg'

    real_image = cv2.imread(real_path)
    real_image_vgg16 = cv2.resize(real_image, (224, 224))
    real_image_resnet50 = cv2.resize(real_image, (224, 224))
    real_image_inception_v3 = cv2.resize(real_image, (299, 299))

    real_image_vgg16 = np.expand_dims(real_image_vgg16, axis=0)
    real_image_resnet50 = np.expand_dims(real_image_resnet50, axis=0)
    real_image_inception_v3 = np.expand_dims(real_image_inception_v3, axis=0)

    real_feature_vgg16 = vgg16.predict(real_image_vgg16)
    real_feature_resnet50 = resnet50.predict(real_image_resnet50)
    real_feature_inception_v3 = inception_v3.predict(real_image_inception_v3)

    real_feature = np.concatenate([real_feature_vgg16.flatten(), real_feature_resnet50.flatten(), real_feature_inception_v3.flatten()])
    features.append(real_feature)

    fake_image = cv2.imread(fake_path)
    fake_image_vgg16 = cv2.resize(fake_image, (224, 224))
    fake_image_resnet50 = cv2.resize(fake_image, (224, 224))
    fake_image_inception_v3 = cv2.resize(fake_image, (299, 299))

    fake_image_vgg16 = np.expand_dims(fake_image_vgg16, axis=0)
    fake_image_resnet50 = np.expand_dims(fake_image_resnet50, axis=0)
    fake_image_inception_v3 = np.expand_dims(fake_image_inception_v3, axis=0)

    fake_feature_vgg16 = vgg16.predict(fake_image_vgg16)
    fake_feature_resnet50 = resnet50.predict(fake_image_resnet50)
    fake_feature_inception_v3 = inception_v3.predict(fake_image_inception_v3)

    fake_feature = np.concatenate([fake_feature_vgg16.flatten(), fake_feature_resnet50.flatten(), fake_feature_inception_v3.flatten()])
    features.append(fake_feature)

features = np.stack(features)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=100)
features_pca = pca.fit_transform(features)

# Perform clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans_labels = kmeans.fit_predict(features_pca)

dbscan = DBSCAN(eps=0.5, min_samples=20)
dbscan_labels = dbscan.fit_predict(features_pca)

# Calculate Silhouette Coefficient for k-means
kmeans_silhouette = silhouette_score(features_pca, kmeans_labels)
print(f"K-means Silhouette Coefficient: {kmeans_silhouette:.3f}")

# Calculate Silhouette Coefficient for DBSCAN
if len(np.unique(dbscan_labels)) > 1:
    dbscan_silhouette = silhouette_score(features_pca, dbscan_labels)
    print(f"DBSCAN Silhouette Coefficient: {dbscan_silhouette:.3f}")
else:
    print("DBSCAN found only one cluster. Silhouette Coefficient cannot be calculated.")

# Plot the clusters
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(features_pca[:, 0], features_pca[:, 1], c=kmeans_labels, cmap='viridis')
ax[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='k', marker='x', s=100, label='Centroids')
ax[0].set_title('K-means Clustering')
ax[0].legend()

ax[1].scatter(features_pca[:, 0], features_pca[:, 1], c=dbscan_labels, cmap='viridis')
ax[1].set_title('DBSCAN Clustering')

plt.tight_layout()
plt.show()