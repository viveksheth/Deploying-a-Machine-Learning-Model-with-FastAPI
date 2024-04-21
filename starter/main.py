# Put the code for your API here.

"""
Script for FastAPI instance and model interface 
Author: Vivek Sheth
Date: Apr 20, 2024

"""

from fastapi import FastAPI, HTTPException
from typing import Union, Optional
import pandas as pd
import os
import pickle
# from ml.data import process_data

# path to saved artifacts
savepath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']

# instantiate FastAPI app
app = FastAPI(title="Inference API",
              description="An API that takes a sample and runs an inference",
              version="1.0.0")

# load model artifacts on startup of the application to reduce latency


@app.get("/")
async def greetings():
    return "Welcome to Fast Model API"


if __name__ == '__main__':
    pass
