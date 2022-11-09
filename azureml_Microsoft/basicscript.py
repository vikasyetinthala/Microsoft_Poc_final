#from azureml.core import Workspace, Experiment, Datastore, Dataset, Run
import pandas as pd 
from sklearn.linear_model import LinearRegression
'''
ws=Workspace.from_config("./config")
az_store= Datastore.get(ws,"azure_sdk_blob01")
az_dataset=Dataset.get_by_name(ws,"")
az_default_store= ws.get_default_datastore()

new_run= Run.get_context() 

df= az_dataset.to_pandas_dataframe()'''

# loading the data in azureml local output  folder

df=pd.DataFrame({"height":[10,20,30,40,50,60,70,80,90],"weights":[100,200,300,400,500,600,700,800,900]})
x=df["height"].values
y=df["weight"].values

lr=LinearRegression() 
lr.fit(x,y)
print('model trained successfully')




