from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle

class DataModeler:
    def __init__(self, sample_df: pd.DataFrame):
        '''
        Initialize the DataModeler as necessary.
        '''
        
        self.train_df = sample_df.copy()
        self.model = None
        
    def prepare_data(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Prepare a dataframe so it contains only the columns to model and having suitable types.
        If the argument is None, work on the training data passed in the constructor.
        '''
        
        if oos_df is None:
            df = self.train_df.copy()
        else:
            df = oos_df.copy()
        
        # Convert the 'transaction_date' column to datetime
        df['transaction_date'] = df['transaction_date'].astype('datetime64[s]')

        # Convert the 'transaction_date' column to float values with handling of NaT values
        df['transaction_date'] = df['transaction_date'].apply(lambda x: x.timestamp() if not pd.isna(x) else np.nan)

        # Convert the 'transaction_date' column to floating-point numbers without truncating decimal values
        df['transaction_date'] = pd.to_numeric(df['transaction_date'])
        
        # Define columns you want to select
        features = ['transaction_date', 'amount', 'outcome']

        # Select features that exist in the DataFrame
        selected_features = [col for col in features if col in df.columns]

        # Create a new DataFrame with the selected features
        df = df[selected_features]

        return df

    def impute_missing(self, oos_df: pd.DataFrame = None) -> pd.DataFrame:
        '''
        Fill any missing values with the appropriate mean (average) value.
        If the argument is None, work on the training data passed in the constructor.
        Hint: Watch out for data leakage in your solution.
        '''
        
        if oos_df is None:
            df = self.train_df.copy()
        else:
            df = oos_df.copy()
            
        # Apply data preparation steps from prepare_data
        df = self.prepare_data(df)
        
        #Define the mean
        train_metrics = self.prepare_data(self.train_df.copy())
        mean_amount = train_metrics['amount'].mean()
        mean_transaction = train_metrics['transaction_date'].mean()
        
        # Impute missing values in amount and transcation date with the mean of the numeric values in the training data
        df['amount'] = df['amount'].fillna(mean_amount)
        df['transaction_date'] = df['transaction_date'].fillna(mean_transaction)

        return df

    def fit(self) -> None:
        '''
        Fit the model of your choice on the training data paased in the constructor, assuming it has
        been prepared by the functions prepare_data and impute_missing
        '''
        
        # Choose a model
        
        
        # Apply imputation of missing values
        train_df= self.impute_missing(self.train_df.copy())
        
        # Fit the model
        self.model = GradientBoostingClassifier().fit(train_df.drop(columns=["outcome"]), train_df["outcome"])


    def model_summary(self) -> str:
        '''
        Create a short summary of the model you have fit.
        '''
        
        
        if self.model is None:
            return "Model has not been trained yet."

        ##  GradientBoostingClassifier model
        if isinstance(self.model, GradientBoostingClassifier):
            feature_importance = self.model.feature_importances_
            params = self.model.get_params()

            summary = f"Gradient Boosting Model Summary:\n"
            summary += f"Feature Importance: {feature_importance}\n"
            summary += f"Model Parameters: {params}\n"
            return summary

        return "Model summary not available for this model type."

    def predict(self, oos_df: pd.DataFrame = None) -> pd.Series[bool]:
        '''
        Make a set of predictions with your model. Assume the data has been prepared by the
        functions prepare_data and impute_missing.
        If the argument is None, work on the training data passed in the constructor.
        '''

        # Make sure the input data has the same columns as the training data
        if oos_df is None:
            df = self.impute_missing(self.train_df.copy())
            #drop target
            df.drop(columns=['outcome'], inplace=True)
        else:
            df = self.impute_missing(oos_df.copy())
            
        predictions = self.model.predict(df)

        return predictions

    def save(self, path: str) -> None:
        '''
        Save the DataModeler so it can be re-used.
        '''
        
        # Salvar o modelo, se necessário
        
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> DataModeler:
        '''
        Reload the DataModeler from the saved state so it can be re-used.
        '''
        
        # Carregar um modelo previamente salvo, se necessário

        with open(path, "rb") as f:
            modeler = pickle.load(f)

        return modeler