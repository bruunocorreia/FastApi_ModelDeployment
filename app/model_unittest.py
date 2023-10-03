#1 library imports
from fastapi import FastAPI
import unittest
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from model import DataModeler
from pathlib import Path

#Class herda uma classe de teste unitário
class TestDataModeler(unittest.TestCase):
    def setUp(self):
        
        # Configurar dados de exemplo para os testes
        self.sample_data = pd.DataFrame({
            "transaction_date": ['2022-01-01', '2022-02-01', None, '2022-03-01'],
            "amount": [1.0, 2.0, 3.0, None],
        })
        
        # Initialize the DataModelerwith training data
        self.modeler = DataModeler(sample_df=pd.DataFrame(
            {
                "customer_id": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                "amount": [1, 3, 12, 6, 0.5, 0.2, np.nan, 5, np.nan, 3],
                "transaction_date": [
                    '2022-01-01',
                    '2022-08-01',
                    None,
                    '2022-12-01',
                    '2022-02-01',
                    None,
                    '2022-02-01',
                    '2022-01-01',
                    '2022-11-01',
                    '2022-01-01'
                ],
                "outcome": [False, True, True, True, False, False, True, True, True, False]
            }
        ))
        
        #Read pickle model
        with open("app\transact_modeler", "rb") as f:
            self.model = pickle.load(f)

    def test_prepare_data(self):
        
        # Create a DataFrame from the input data
        input_df = self.sample_data
        
        # Preprocess
        prepared_data = self.modeler.prepare_data(input_df)
        
        # Verifique se os dados preparados têm as colunas esperadas
        expected_columns = ['transaction_date', 'amount']
        self.assertEqual(list(prepared_data.columns), expected_columns)

        # Verifique se os valores de 'transaction_date' foram convertidos corretamente
        self.assertTrue(pd.api.types.is_float_dtype(prepared_data['transaction_date']))
        
        # Verifique se os valores de 'amount' foram convertidos corretamente
        self.assertTrue(pd.api.types.is_float_dtype(prepared_data['amount']))

    def test_impute_missing(self):
        
        # Create a DataFrame from the input data
        input_df = self.sample_data
        
        # Impute missing values
        imputed_data = self.modeler.impute_missing(input_df)

        # Verifique se os valores ausentes foram imputados corretamente (usando a média)
        self.assertEqual(imputed_data['amount'].isna().sum(), 0)
        self.assertEqual(imputed_data['transaction_date'].isna().sum(), 0)    
    
    
    def test_scoring(self):
        
        # Create a DataFrame from the input data
        input_df = self.sample_data
        
        # Impute missing values
        imputed_data = self.modeler.impute_missing(input_df)
        
        # Score
        scoring = self.model.predict(imputed_data)
        
        # Verifique se as previsões são de nulo
        self.assertTrue(all(pred is not None for pred in scoring))
        
if __name__ == '__main__':
    unittest.main()
