import numpy as np
import cv2
import matplotlib.pyplot as plt

np.random.seed(276)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KMeans:
    def __init__(self, K=3, max_iters=100):
        self.K = K
        self.max_iters = max_iters

    def predict(self, X,image_shape = None):
        self.X = X
        self.image_shape = image_shape
        self.n_samples, self.n_features = X.shape

        # Initialize centroids randomly
# Assuming self.X is reshaped from the image with shape (h*w, 3)
# You need to get indices for 3 selected pixels
        h, w = self.image_shape  # Save this before calling predict()

# Compute 1D indices from 2D coordinates
        def to_index(row, col):
           return row * w + col

        idx_corner = to_index(0, 0)
        idx_center = to_index(h // 2, w // 2)
        idx_between = to_index(h // 4, w // 4)

        self.centroids = [self.X[idx_corner], self.X[idx_center], self.X[idx_between]]


        for _ in range(self.max_iters):
            clusters = self._create_clusters(self.centroids)
            new_centroids = self._get_centroids(clusters)

            if self._is_converged(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self._get_labels(clusters)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            distances = [euclidean_distance(sample, point) for point in centroids]
            closest_idx = np.argmin(distances)
            clusters[closest_idx].append(idx)
        return clusters

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for i, cluster in enumerate(clusters):
            if cluster:
                centroids[i] = np.mean(self.X[cluster], axis=0)
            else:
                centroids[i] = self.X[np.random.choice(self.n_samples)]
        return centroids

    def _is_converged(self, old, new):
        return np.allclose(old, new)

    def _get_labels(self, clusters):
        labels = np.empty(self.n_samples, dtype=int)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

def main():
    # Load and convert to RGB
    img = cv2.imread("12.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, c = img.shape
    img_flat = img.reshape(-1, 3)

    # Run KMeans with 3 clusters
    kmeans = KMeans(K=3)
    labels = kmeans.predict(img_flat,image_shape=(h,w))

    # Map clusters to white, white, green
    colors = np.array([
        [255, 255, 255],  # white
        [0, 255, 0],      # green
        [255, 255, 255]       # white
    ], dtype=np.uint8)

    clustered_img = np.array([colors[label] for label in labels])
    clustered_img = clustered_img.reshape(h, w, c)

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(clustered_img)
    plt.title("Clustered (White, White, Green)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    cv2.imwrite("clustered_output.png", clustered_img)


if __name__ == "__main__":
    main()
