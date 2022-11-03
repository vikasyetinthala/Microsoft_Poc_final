import joblib 
import json 
from azureml.core.model import Model 
import pandas as pd
import time

def init():
    global ref_cols, predictor 
    model_path=Model.get_model_path('MS_POC_Titanic_Model')
    ref_cols, predictor = joblib.load(model_path)

def run(raw_data):
    try:
        data_dict=json.loads(raw_data)['data']
        data= pd.DataFrame.from_dict(data_dict)
        data_enc=pd.get_dummies(data)
        deploy_cols=data_enc.columns
        missing_cols=ref_cols.difference(deploy_cols)
        for col in missing_cols:
            data_enc[col]=0
        data_enc=data_enc[ref_cols]
        predictions = predict(data_enc)
        classes=[0,1]
        predicted_classes=[]
        for prediction in predictions:
            predicted_classes.append("......")
        return json.dumps(predicted_classes)
    except Exception as e:
        error=str(e)
        print(error+time.strftime("%H:%M:%S"))
        return error