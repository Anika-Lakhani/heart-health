from src.data.data_loader import DataLoader
from src.data.feature_engineering import FeatureEngineer
from src.models.clustering import PatientClustering
from src.visualization.plotting import ClusterVisualizer
import config
import pandas as pd

def main():
    # Initialize classes
    loader = DataLoader(config.DATA_DIR)
    engineer = FeatureEngineer()
    clustering = PatientClustering()
    visualizer = ClusterVisualizer()
    
    # Load data
    data_files = loader.load_csv_files()
    
    # Process features
    bp_features = engineer.process_blood_pressure(data_files['observations'])
    demographics = engineer.process_demographics(data_files['patients'])
    
    # Combine features
    combined_features = pd.merge(bp_features, demographics, 
                               left_index=True, right_on='Id')
    
    # Prepare for clustering
    features = clustering.prepare_features(combined_features, 
                                        config.NUMERICAL_FEATURES)
    
    # Find optimal clusters
    optimal_k, silhouette_scores = clustering.find_optimal_clusters(features)
    
    # Perform final clustering
    cluster_labels = clustering.fit_predict(features, optimal_k)
    
    # Visualize results
    visualizer.plot_cluster_distributions(combined_features, cluster_labels, 
                                       config.NUMERICAL_FEATURES)
    visualizer.plot_cluster_scatter(combined_features, cluster_labels, 
                                  'mean_systolic', 'mean_diastolic')
    
    # Save processed data
    combined_features['cluster'] = cluster_labels
    combined_features.to_csv(config.PROCESSED_DIR / 'processed_data.csv', 
                           index=False)

if __name__ == "__main__":
    main()