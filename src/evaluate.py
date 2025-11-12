import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import pickle

from sklearn.metrics import accuracy_score

import sys


df=pd.read_csv('data/IRIS.csv')

x=df.drop(columns=['species'])

y=df['species']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

with open("models/model.pkl","rb") as f:

  model=pickle.load(f)

y_pred=model.predict(X_test)

acc=accuracy_score(y_test,y_pred)

print(f"accuracy score={acc:.4f}")

if acc<0.95:

  print("accuracy below threshold,exit pipeline")

  sys.exit(1)

