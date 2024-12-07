# Heart Health Project

A data analysis and prediction project focused on heart health using synthetic patient data.

## Project Structure

### Core Components

- `notebooks/` - Jupyter notebooks for data analysis and visualization
  - Exploratory data analysis
  - Model development
  - Results visualization

- `output/` - Synthetic data results from Synthea that will be clustered
  - CSV format
  - fhir format
  - metadata

- `src/` - Source code directory containing core functionality
  - `data/`
    - `data_loader.py` - Handles FHIR and CSV data import, data cleaning, and preprocessing
    - `feature_engineering.py` - Creates derived features from raw patient data
  - `models/`
    - `clustering.py` - Implements K-means and hierarchical clustering algorithms
    - `risk_score.py` - We need to make this to generate a risk score for each patient
  - `visualization/`
    - `plotting.py` - Visualization utilities for clusters

#### Data Processing (`src/data/`)
- **Data Loader**: 
  - Supports both FHIR (JSON) and CSV formats
  - Handles missing data imputation
  - Normalizes dates and numerical values
  - Performs basic data validation and cleaning

- **Feature Engineering**: (need to check over this description: AI-generated)
  - Temporal feature extraction (e.g., BP variation over time)
  - BMI calculation and categorization
  - Risk factor aggregation
  - Lab result normalization
  - Medication history vectorization

#### Analysis (`src/models/`)
- **Clustering Pipeline**: (need to check over this description: AI-generated)
  - Patient similarity computation
  - Dimensionality reduction using PCA/t-SNE
  - K-means clustering for patient subgroup identification
  - Hierarchical clustering for risk pattern discovery

- **Visualization Components**: (need to check over this description: AI-generated)
  - Interactive cluster visualization using plotly
  - Patient trajectory plots
  - Feature importance charts
  - Risk distribution histograms
  - Temporal trend analysis plots

### Configuration Files

- `config.py` - Project configuration settings
- `requirements.txt` - Python package dependencies
- `characteristics.txt` - Dataset feature definitions
- `synthea-instructions.md` - Instructions for synthetic data generation

## Setup and Installation

1. Create a virtual environment:

python
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate

2. Install dependencies:

pip install -r requirements.txt

3. Use Synthea to generate synthetic data or use the 'results' folder.

4. Run 'python main.py' to process the data and train the model.