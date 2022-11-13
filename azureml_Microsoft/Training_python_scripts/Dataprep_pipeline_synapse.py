from azureml.core import Run, Dataset, Datastore
import pandas as pd 
import numpy as np
import os 
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--datafolder",type=str)
args= parser.parse_args()
new_run=Run.get_context() 
ws=new_run.experiment.workspace 
df= new_run.input_datasets["raw_data"].to_pandas_dataframe()
total_observations=len(df)
df=df[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Survived"]]

df["Age"]=df["Age"].fillna(np.mean(df["Age"]))
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].value_counts(ascending=False).index[0])
df=pd.get_dummies(df)
cols=df.select_dtypes(include='object').columns
#os.makedirs(args.datafolder,exist_ok=True)
#path= os.path.join(args.datafolder,'defaults_prep.csv')
#df.to_csv(args.datafolder,index=False)
az_default_store= ws.get_default_datastore()
Dataset.Tabular.register_pandas_dataframe(df, az_default_store, "azureml_processed_dataset_synapse", show_progress=True)
new_run.log("Total Observations: ",total_observations)

new_run.complete()