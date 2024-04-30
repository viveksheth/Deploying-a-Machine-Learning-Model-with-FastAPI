"""
Script to test FastAPI instance for model inference
Author: Vivek Sheth
Date: April 23, 2024
"""

import requests
import json

url = "https://deploying-a-machine-learning-model-with.onrender.com/inference/"

# explicit the sample to perform inference on
sample = {'age': 52,
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


def sample_data(sample_data):
    return json.dumps(sample_data)


def test_api():
    try:
        data = sample_data(sample)
        # post to API and collect response
        response = requests.post(url, data=data)
        # display output - response will show sample details + model prediction added
        response.raise_for_status()
        print("response status code", response.status_code)
        print("response content:")
        print(response.json())
    except requests.exceptions.RequestException as reqException:
        print("Error occurred during API test: ", reqException)
        return None


if '__name__' == '__main__':
    test_api()
