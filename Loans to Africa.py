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

##To identify whether dataset is 'Series' or 'Dataframe'
print(type(dataset))

## list with specific columns relevant to data analysis
list1 = ["Year", "Country", "USD (M)", "Interest Rate", "Libor Rate", "Term", "Sector"]
print(dataset[list1].head(10))

##Dropping columns that contain null values
dropcolumns = dataset.dropna(axis=1)
print(dataset.shape,dropcolumns.shape)

##Replacing null columns with zero
cleareddata = dataset.fillna(0)
print(cleareddata.isnull().sum())

##Name of individual countries in the dataset
print(dataset.Country.unique())

##Summary Statistics
Amount = dataset['USD (M)']
print(Amount.sum())

print(Amount.max())

print(Amount.mean())

print(Amount.median())

## Visualisations (Seaborn scatterplot)
sns.set_style('darkgrid')
sns.relplot(data=dataset,x='USD (M)', y='Country')

plt.title("Country vs Loan Amount")
plt.xlabel("USD (M)")
plt.ylabel("Country")
plt.show()










