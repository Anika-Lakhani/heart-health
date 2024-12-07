from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd

class PatientClustering:
    def __init__(self, n_clusters_range=(2, 11)):
        self.n_clusters_range = range(*n_clusters_range)
        self.scaler = StandardScaler()
        self.pca = None
        self.best_model = None
        
    def prepare_features(self, df: pd.DataFrame, numerical_features: list) -> np.ndarray:
        """Standardize and reduce dimensionality of features"""
        scaled_features = self.scaler.fit_transform(df[numerical_features])
        
        # Apply PCA
        self.pca = PCA(n_components=0.95)
        reduced_features = self.pca.fit_transform(scaled_features)
        
        return reduced_features
        
    def find_optimal_clusters(self, features: np.ndarray) -> tuple:
        """Find optimal number of clusters using silhouette score"""
        silhouette_scores = []
        
        for k in self.n_clusters_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(features)
            score = silhouette_score(features, clusters)
            silhouette_scores.append(score)
            
        optimal_k = self.n_clusters_range[np.argmax(silhouette_scores)]
        return optimal_k, silhouette_scores
        
    def fit_predict(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        """Fit final clustering model"""
        self.best_model = KMeans(n_clusters=n_clusters, random_state=42)
        return self.best_model.fit_predict(features)