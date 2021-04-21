import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
dataset = pd.read_csv("ChinaLoansToAfrica.csv")

##Missing values count
print(dataset.isnull().sum())

## To check the shape of the  i.e. number of rows and columns
print(dataset.shape)

## To know the names of the columns
print(dataset.columns)

##To check the number of unique identifiers or primary key
print(len(dataset['Loan ID']))

##To check the type of data in each column
print(dataset.dtypes)

##To see the unique years
print(dataset.Year.unique())

##To see how many times a year appears in the dataset
print(dataset['Year'].value_counts(2017))














