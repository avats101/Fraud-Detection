import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model #Importing Linear model from sklearn library
from sklearn.model_selection import train_test_split #Importng train_test_split to split the data for training and testing.
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
import warnings #just to ignore unnecessary warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("creditcard2.csv") #Reading the dataset

#Removing the unnecessary data columns nameDest,nameOrig,type
del data['nameDest']
del data['nameOrig']
del data['type']

X = data[['step','amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest','isFlaggedFraud']]# Here is X are the Features 
y = data['isFraud'] # Here is y are Target
logitic = linear_model.LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = logitic.fit(X_train,y_train)

predictions = model.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test, predictions))
accuracy_score(y_test,predictions)
print ("Logistic Regression Accuracy", accuracy_score(y_test,predictions))
