import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ClusterVisualizer:
    def __init__(self, save_dir: str = 'results/figures'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_cluster_distributions(self, df, cluster_labels, features, 
                                 filename='cluster_distributions.png'):
        """Plot distribution of features across clusters"""
        df['Cluster'] = cluster_labels
        
        fig, axes = plt.subplots(len(features), 1, figsize=(12, 4*len(features)))
        for i, feature in enumerate(features):
            sns.boxplot(data=df, x='Cluster', y=feature, ax=axes[i])
            axes[i].set_title(f'{feature} Distribution by Cluster')
            
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()
        
    def plot_cluster_scatter(self, df, cluster_labels, x, y, 
                           filename='cluster_scatter.png'):
        """Create scatter plot of clusters"""
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x=x, y=y, hue=cluster_labels, 
                       palette='deep')
        plt.title(f'Patient Clusters: {x} vs {y}')
        plt.savefig(self.save_dir / filename)
        plt.close()