from azureml.core import Workspace, Experiment, Datastore, Dataset, Run
import pandas as pd 
 
ws=Workspace.from_config("./config")
az_store= Datastore.get(ws,"azure_sdk_blob01")
az_dataset=Dataset.get_by_name(ws,"")
az_default_store= ws.get_default_datastore()

new_run= Run.get_context() 

df= az_dataset.to_pandas_dataframe()

# loading the data in azureml local output  folder

df=df[["col1","col2"]]
df.to_csv("./outputs/newdata.csv",index=False)

total_observations=len(df)
null_df=df.isnull().sum()
new_run.log("Total observations: ",total_observations)
for columns in df.columns:
    new_run.log(columns,nulldf[columns])
# loading the data in azureml local output  folder

new_run.complete() 
