from azureml.core import Run, Dataset
import pandas as pd
import numpy as np
import joblib 
import json
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler

new_run = Run.get_context(allow_offline=False) 
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(AzureLogHandler(connection_string='InstrumentationKey=beae4d86-caa3-4284-9a1b-8285f65c4b59;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/'))
custom_dimensions_score={
        "parent_run_id": new_run.parent.id,
        "step_id":new_run.id,
        "step_name":new_run.name,
        "experiment_name":new_run.experiment.name,
        "run_url":new_run.parent.get_portal_url(),
        "run_type":"training"
    }
ws=new_run.experiment.workspace 
import argparse 
parser= argparse.ArgumentParser()
parser.add_argument("--n_estimators",type=int)
parser.add_argument("--min_samples_leaf",type=int)
parser.add_argument("--input-data",type=str)

args=parser.parse_args()

ne =args.n_estimators 
ms1= args.min_samples_leaf



try:
    az_dataset = Dataset.get_by_name(ws,"Titanic_vikas_logging")

    df = az_dataset.to_pandas_dataframe()
    df=df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]
    missing=dict(df.isna().sum())
    missing["runid"]=new_run.parent.id
    '''
    logger.warning("Missing values are",extra=missing)
    '''
    df["Age"]=df["Age"].fillna(np.mean(df["Age"]))
    df["Embarked"]=df["Embarked"].fillna(df["Embarked"].value_counts(ascending=False).index[0])
    df=pd.get_dummies(df)
    x=df.drop(['Survived'],axis=1)
    #y=dataprep["Survived"]
    y=pd.DataFrame(df["Survived"],columns=["Survived"])
    train_enc_cols= x.columns
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)

    from sklearn.ensemble import RandomForestClassifier
    rfc=RandomForestClassifier(n_estimators=ne,min_samples_leaf=ms1)
    trained_model=rfc.fit(x_train,y_train)

    pred=rfc.predict(x_test)

    y_prob=rfc.predict_proba(x_test)[:,1]

    from sklearn.metrics import confusion_matrix
    cm=confusion_matrix(y_test,pred)
    score=rfc.score(x_test,y_test)
    model_file='./outputs/models.pkl'
    joblib.dump(value=[train_enc_cols,trained_model],filename=model_file)

    new_run.log("Accuracy ",score)
    logger.info("model trained successfully",extra=custom_dimensions_score)
except Exception as e:
    logger.error(e,extra=custom_dimensions_score)

new_run.complete()

