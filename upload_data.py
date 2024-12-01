from pymongo.mongo_client import MongoClient
import pandas as pd
import json
uri="mongodb+srv://hemant:Hemant_31@cluster0.uz1jk.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
#create a new client and connectt to server
client = MongoClient(uri)
#create database name and collection name
DATABASE_NAME="sensor"
COLLECTION_NAME='waferfault'
df=pd.read_csv(r"D:/sensor_pro/notebook/wafer_23012020_041211.csv")
df.head()
json_record=list(json.loads(df.T.to_json()).values())
client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)