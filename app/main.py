#1 library imports
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from app.model import DataModeler
from pathlib import Path

#We use to save the version of our softwar
__version__ = "0.1.0"

#Read pickle model
with open("transact_modeler", "rb") as f:
    model = pickle.load(f)

# Initialize the DataModelerwith training data
modeler = DataModeler(sample_df=pd.DataFrame(
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
                                                    "outcome" : [False, True, True, True, False, False, True, True, True, False]
                                                }
                                            ))
#2 create the app object
app = FastAPI()

#Define my class of input data (this help us to implement every thing and test type of the input before)
class InputData(BaseModel):
    transaction_date: float
    amount: float

#index toute
@app.get('/')
def index():
    return {'message': 'hello, world'}

@app.post("/predict")
#async para permitir funcao assincrona
async def predict_outcome(data: InputData):
    
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([data.dict()])

    #preprocess
    prepared_data = modeler.impute_missing(input_df)
    
    # Make predictions
    predictions = model.predict(prepared_data)

    # Return the prediction as a JSON response
    return {"prediction": bool(predictions[0])}
              
#5 Run the API with univicorn
if __name__ =='__main__':
    uvicorn.run(app, host ='127.0.0.1', port=8508)

#uvicorn main:app --reload
#uvicorn offerfit_main_fastAPI:app --reload

