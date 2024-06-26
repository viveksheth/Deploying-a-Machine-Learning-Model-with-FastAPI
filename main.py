"""
Code for FastAPI
Author: Vivek Sheth
Date: Apr 20, 2024
"""

from fastapi import FastAPI
import pandas as pd
import os
import pickle
from pydantic import BaseModel
from ml.data import process_data

# path to saved artifacts
savepath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                'age': 50,
                'workclass': "Private",
                'fnlgt': 234721,
                'education': "Doctorate",
                'education_num': 16,
                'marital_status': "Separated",
                'occupation': "Exec-managerial",
                'relationship': "Not-in-family",
                'race': "Black",
                'sex': "Female",
                'capital_gain': 0,
                'capital_loss': 0,
                'hours_per_week': 50,
                'native_country': "United-States"
            }
        }


# instantiate FastAPI app
app = FastAPI(title="Inference API",
              description="An API that takes a sample and runs an inference",
              version="1.0.0")


@app.get("/")
async def read_main():
    return {"message": "Welcome to Fast API"}


@app.post("/inference/")
async def ingest_data(inference: InputData):
    data = {'age': inference.age,
            'workclass': inference.workclass,
            'fnlgt': inference.fnlgt,
            'education': inference.education,
            'education-num': inference.education_num,
            'marital-status': inference.marital_status,
            'occupation': inference.occupation,
            'relationship': inference.relationship,
            'race': inference.race,
            'sex': inference.sex,
            'capital-gain': inference.capital_gain,
            'capital-loss': inference.capital_loss,
            'hours-per-week': inference.hours_per_week,
            'native-country': inference.native_country,
            }

# prepare the sample for inference as a dataframe
    sample = pd.DataFrame(data, index=[0])

    # apply transformation to sample data
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # if saved model exits, load the model from disk
    if os.path.isfile(os.path.join(savepath, filename[0])):
        model = pickle.load(open(os.path.join(savepath, filename[0]), "rb"))
        encoder = pickle.load(open(os.path.join(savepath, filename[1]), "rb"))
        lb = pickle.load(open(os.path.join(savepath, filename[2]), "rb"))

    sample, _, _, _ = process_data(
        sample,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # get model prediction
    prediction = model.predict(sample)

    # convert prediction to label and add to data output
    if prediction[0] > 0.5:
        prediction = '>50K'
    else:
        prediction = '<=50K',
    data['prediction'] = prediction

    return data

if __name__ == '__main__':
    pass
