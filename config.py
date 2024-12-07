from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'output' / 'csv'
RESULTS_DIR = BASE_DIR / 'results'
PROCESSED_DIR = RESULTS_DIR / 'processed'
FIGURES_DIR = RESULTS_DIR / 'figures'

# Create directories if they don't exist
for dir_path in [PROCESSED_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Feature configurations
NUMERICAL_FEATURES = [
    'mean_systolic',
    'mean_diastolic',
    'mean_hdl',
    'mean_ldl',
    'mean_total_cholesterol',
    'mean_glucose',
    'age_at_death'
]

CATEGORICAL_FEATURES = [
    'GENDER',
    'RACE',
    'diabetes_history',
    'smoking_status',
    'hypertension_treatment'
]