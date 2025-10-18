import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import  GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(x, y, models, params):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv= 3)
            gs.fit(x, y)

            model.set_params(**gs.best_params_)
            model.fit(x, y)
            y_pred = model.predict(x)
            report[list(models.keys())[i]] = r2_score(y, y_pred)
            
        return report
    except Exception as e:
        raise CustomException(e, sys)