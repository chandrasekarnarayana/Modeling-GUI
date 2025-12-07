import numpy as np
from sklearn.datasets import make_blobs

from modeling_gui.models import ModelManager


def test_kmeans_basic_clusters_and_centers():
    X, _ = make_blobs(n_samples=60, centers=3, n_features=2, random_state=42, cluster_std=0.5)
    mgr = ModelManager()
    model = mgr.kmeans_clustering(X, n_clusters=3)
    assert model.n_clusters == 3
    labels = model.labels_
    assert len(labels) == len(X)
    centers = model.cluster_centers_
    assert centers.shape == (3, 2)


def test_kmeans_respects_n_clusters():
    X, _ = make_blobs(n_samples=30, centers=4, n_features=2, random_state=0, cluster_std=0.3)
    mgr = ModelManager()
    model = mgr.kmeans_clustering(X, n_clusters=4)
    assert model.n_clusters == 4
    assert model.cluster_centers_.shape[0] == 4
