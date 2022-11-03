from azureml.core import Run, Experiment, Workspace
import pandas as pd 
import numpy as np
import os 
import argparse
import joblib
import json
from azureml.interpret import ExplanationClient 

parser=argparse.ArgumentParser()
parser.add_argument("--datafolder",type=str)
args= parser.parse_args() 
new_run=Run.get_context() 
ws=new_run.experiment.workspace 

path= os.path.join(args.datafolder, 'defaults_prep.csv')

dataprep= pd.read_csv(path)

x=dataprep.drop(['Survived'],axis=1)
#y=dataprep["Survived"]
y=pd.DataFrame(dataprep["Survived"],columns=["Survived"])
train_enc_cols= x.columns

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=123)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
trained_model=lr.fit(x_train,y_train)

pred=lr.predict(x_test)

y_prob=lr.predict_proba(x_test)[:,1]

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
score=lr.score(x_test,y_test)
from interpret_community.tabular_explainer import TabularExplainer
classes=[0,1]
features = list(x.columns)
tab_explainer = TabularExplainer(trained_model,
                                x_train,
                                features=features,
                                classes=classes)
global_explanation= tab_explainer.explain_global(x_train)
global_fi = global_explanation.get_feature_importance_dict()
from azureml.interpret import ExplanationClient 
explain_client= ExplanationClient.from_run(new_run)
explain_client.upload_model_explanation(global_explanation,comment="My First Explanation")


'''
cm_dict={"schema_type":"confusion_matrix",
        "schema_version":"v1",
        "data":{"class_labels":[0,1],
        "matrix":cm.tolist()}
        }

new_run.log_confusion_matrix("ConfusionMatrix",cm_dict)'''
new_run.log("score ",score)
'''
x_test=x_test.reset_index(drop=True)
y_test=y_test.reset_index(drop=True)
y_prob_df=pd.DataFrame(y_prob,columns=["Scored Probabilities"])
y_predict_df=pd.DataFrame(pred,columns="Scored_Label")
scored_dataset=pd.concat([x_test,y_test,y_prob_df,y_predict_df],axis=1)

scored_dataset.to_csv("./outputs/defaults_scored.csv")'''
model_file='./outputs/models.pkl'
joblib.dump(value=[train_enc_cols,trained_model],filename=model_file)
new_run.complete()















