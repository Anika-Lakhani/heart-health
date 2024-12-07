import pandas as pd
import numpy as np
from typing import Dict

class FeatureEngineer:
    # LOINC codes for different measurements
    CODES = {
        'blood_pressure': {
            'systolic': '8480-6',
            'diastolic': '8462-4'
        },
        'cholesterol': {
            'hdl': '2085-9',
            'ldl': '2089-1',
            'total': '2093-3'
        },
        'glucose': '2339-0'
    }

    def process_blood_pressure(self, observations: pd.DataFrame) -> pd.DataFrame:
        bp_data = observations[observations['CODE'].isin(self.CODES['blood_pressure'].values())]
        
        # Calculate statistics for each patient
        bp_features = bp_data.groupby(['PATIENT', 'CODE']).agg({
            'VALUE': ['mean', 'std', 'min', 'max'],
            'DATE': ['first', 'last']
        }).reset_index()
        
        # Pivot to create separate columns for systolic/diastolic
        bp_features = bp_features.pivot(index='PATIENT', 
                                      columns='CODE', 
                                      values=('VALUE', 'DATE'))
        
        return bp_features

    def process_demographics(self, patients: pd.DataFrame) -> pd.DataFrame:
        demographics = patients.copy()
        demographics['AGE_AT_DEATH'] = pd.to_datetime(demographics['DEATHDATE']) - \
                                     pd.to_datetime(demographics['BIRTHDATE'])
        demographics['AGE_AT_DEATH'] = demographics['AGE_AT_DEATH'].dt.total_days() / 365.25
        
        return demographics[['Id', 'GENDER', 'RACE', 'AGE_AT_DEATH']]

    # Add other processing methods...