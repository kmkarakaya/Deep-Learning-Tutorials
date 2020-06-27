# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 13:47:05 2020
MURAT KARAKAYA AKADEMİ

NASIL SUNSAM

@author: kmkarakaya
"""
# %%
# LOAD MODEL
from joblib import load

filename="myFirstSavedModel.joblib"

clfUploaded = load(filename)
# %%
from sklearn.datasets import load_iris
dataSet = load_iris()
labelsNames = list(dataSet.target_names)


# %%

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import numpy as np


templates = Jinja2Templates(directory="templates")

app = FastAPI()


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})




@app.get("/predict")
async def make_prediction(request: Request, L1:float, W1:float,
                          L2:float, W2:float):
    testData= np.array([L1,W1,L2,W2]).reshape(-1,4)
    probalities = clfUploaded.predict_proba(testData)[0]
    predicted = np.argmax(probalities)
    probabilty= probalities[predicted]
    predicted = labelsNames [predicted]
    return templates.TemplateResponse("prediction.html",
                                      {"request": request,
                                       "probalities": probalities,
                                      "predicted": predicted,
                                      "probabilty": probabilty}  )







