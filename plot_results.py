import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_data(embeddings_path, labels_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return embeddings, labels

def plot_3d(embeddings, labels, output_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], embeddings[:, 2], c=labels, cmap='viridis', s=50, alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    ax.set_title('3D Visualization of HDBSCAN Clustering')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    plt.savefig(output_path)
    plt.show()

if __name__ == "__main__":
    embeddings_path = './results/method_embeddings.pkl'
    labels_path = './results/method_labels.pkl'
    output_path = './results/hdbscan_clusters_3d.png'

    embeddings, labels = load_data(embeddings_path, labels_path)
    plot_3d(embeddings, labels, output_path)
