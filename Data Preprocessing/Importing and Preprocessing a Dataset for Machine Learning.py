# Importing the necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd 

# Loading the Iris dataset
dataset=pd.read_csv('iris.csv')
# Creating the matrix of features (X) and the dependent variable vector (y)
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
# Printing the matrix of features and the dependent variable vector
print(X)
print(y)
