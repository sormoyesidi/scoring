### Script de conception de l'API pour la Prédiction et les features importance
### Auteur: Kanata BEN BARKA

import shap
import json
import joblib
import pandas as pd
import numpy as np
import uvicorn
import lightgbm
import imblearn
import gc
from fastapi import FastAPI


# Creation of the API
app = FastAPI()

@app.get("/")
async def root():
    return{'message':'API Credit Scoring'}

df = pd.read_csv('./Data/test_api.csv', index_col='SK_ID_CURR')
df.drop(['Unnamed: 0','Unnamed: 0.1','TARGET'], axis=1, inplace= True)
feats =  list(df.columns)
model = joblib.load('model_sans_seuil.sav')
clf = model['classifier']
gc.collect()


def score_proba(proba):
    if proba >= 0.5:
        score = 'D'
    elif 0.5 > proba >= 0.424:
        score = 'C'
    elif 0.424 > proba >= 0.30:
        score = 'B'
    else:
        score = 'A'
    return score


# Prédiction
@app.get("/prediction")
async def prediction(id : int):
    #Chargement du modèle, prédiction avec seuil de prédiction et score
    X = pd.DataFrame(df.loc[id, :]).T
    proba = clf.predict_proba(X)[:,1][0]
    pred = (proba > 0.424)
    pred = np.multiply(pred, 1)
    score = score_proba(proba)
    pred = int(pred)
    score = str(score)
    result = {'prediction': pred, 'probabilité':proba, 'score': score}
    del proba, pred, score
    gc.collect()
    return result

#SHAP features local
@app.get("/feat_local")
def feat_local(id:int):
    #Fonction qui extrait et renvoie les features importance locales
    X = pd.DataFrame(df.loc[id, :]).T
    explainer = shap.Explainer(clf,df)
    shap_local_val = explainer(X)
    features_shap = {}
    for n in range (-1,-11,-1):
        arg = abs(shap_local_val.values.mean(axis=0)).argsort()[n]
        df_temp = pd.DataFrame(feats)
        keys = df_temp.loc[arg,:].values
        key = keys.astype('str').tolist()[0]
        val = shap_local_val.values.mean(axis=0)[arg]
        features_shap[key] = val
    del X, shap_local_val, arg, df_temp, keys, key, val
    gc.collect()
    return features_shap

#SHAP features globale
@app.get("/feat_glob")
def feat_glob():
    #Fonction qui extrait et renvoie les features importance globales
    explainer = shap.Explainer(clf,df)
    shap_val = explainer(df)
    features_shapey = {}
    for n in range (-1,-11,-1):
        arg = abs(shap_val.values.mean(axis =0)).argsort()[n]
        df_temp = pd.DataFrame(feats)
        keys = df_temp.loc[arg,:].values
        key = keys.astype('str').tolist()[0]
        val = abs(shap_val.values.mean(axis =0))[arg]
        features_shapey[key] = val
    return features_shapey
