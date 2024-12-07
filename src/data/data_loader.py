import pandas as pd
from pathlib import Path
from typing import Dict, List

class DataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """Load all relevant CSV files from Synthea output"""
        files = {
            'observations': pd.read_csv(self.data_path / 'observations.csv'),
            'patients': pd.read_csv(self.data_path / 'patients.csv'),
            'conditions': pd.read_csv(self.data_path / 'conditions.csv'),
            'medications': pd.read_csv(self.data_path / 'medications.csv')
        }
        return files

    @staticmethod
    def get_code_values(df: pd.DataFrame, codes: List[str]) -> pd.DataFrame:
        """Extract values for specific LOINC/SNOMED codes"""
        return df[df['CODE'].isin(codes)]