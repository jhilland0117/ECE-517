import numpy as np
import matplotlib.pyplot as plt

def generate_gaussian_cluster(center, variance, num_samples):
    cov_matrix = np.identity(2) * variance
    return np.random.normal(center, cov_matrix, num_samples)

def main():
    # Parameters
    num_samples_per_cluster = 10
    variance = 0.1

    # Generate samples for each cluster
    cluster1 = generate_gaussian_cluster([1, 1], variance, num_samples_per_cluster)
    cluster2 = generate_gaussian_cluster([2, 1.5], variance, num_samples_per_cluster)
    cluster3 = generate_gaussian_cluster([2, 1], variance, num_samples_per_cluster)
    cluster4 = generate_gaussian_cluster([3, 1.5], variance, num_samples_per_cluster)

    # Combine clusters
    all_samples = np.vstack([cluster1, cluster2, cluster3, cluster4])

    # Assign labels
    labels = np.ones(num_samples_per_cluster * 2)  # 1 for the first two clusters
    labels = np.concatenate([labels, -11 * np.ones(num_samples_per_cluster * 2)])  # -11 for the rest

    # Plot the generated samples
    plt.scatter(all_samples[:, 0], all_samples[:, 1], c=labels, cmap='viridis')
    plt.title('Generated 2D Samples in Four Gaussian Clusters')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

if __name__ == "__main__":
    main()