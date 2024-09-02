# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# Load the dataset
dataset = pd.read_csv('titanic.csv')
# Identify the categorical data
categorical_features = ['Sex', 'Embarked', 'Pclass']
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),categorical_features)],remainder='passthrough')
X = ct.fit_transform(dataset)
X = np.array(X)
# Implement an instance of the ColumnTransformer class
le = LabelEncoder()
y = le.fit_transform(dataset['Survived'])
print( X)
print( y)
