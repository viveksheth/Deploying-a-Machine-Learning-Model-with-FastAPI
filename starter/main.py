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
from ml.data import process_data

# path to saved artifacts
savepath = './model'
filename = ['trained_model.pkl', 'encoder.pkl', 'labelizer.pkl']
