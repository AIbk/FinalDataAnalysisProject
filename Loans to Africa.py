
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv("ChinaLoansToAfrica.csv")

dataset2 = pd.read_excel('African Countries GDP.xlsx')
print(dataset2.shape)

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

##Dropping columns that contain null values in dataset2
print(dataset2.shape,dropcolumns.shape)


##Replacing null columns with zero
cleareddata = dataset.fillna(0)
print(cleareddata.isnull().sum())

##Name of individual countries in the dataset
print(dataset.Country.unique())

## Sorting
print(dataset2['Country'].sort_values())

## Sorting for Countries with highest GDP
SortbyGDP=(dataset2.sort_values(by=['Nominal GDP ($billions)'], ascending=False))
print(SortbyGDP)

##Top eleven countries by GDP
print(SortbyGDP.head(11))

## Data Merging
Mergeddata = (pd.merge(dataset,dataset2, on="Country"))
print(Mergeddata)


##Summary Statistics
Amount = dataset['USD (M)']
print(Amount.sum())

print(Amount.max())

print(Amount.mean())

print(Amount.median())


##Numpy calculations
import numpy as np
print(np.sum(Amount))
print(np.max(Amount))
print(np.min(Amount))
print(np.mean(Amount))

##sorting with numpy
a=np.array([2019,2017,2015,2016])
print(np.sort(a))


##Looping with 'Break' statement
loans =[23787,388,77,43,2002,1108]
for loans in loans:
    if loans == 77:
        print("I Found it!")
        'Break'
        print(loans)

##Looping with 'Continue' statement
loans =[23787,388,77,43,2002,1108]
for loans in loans:
    if loans == 77:
        print("I Found it!")
        'Continue'
        print(loans)


## Visualisation (Matplotlib)
import matplotlib.pyplot as plt
plt.subplots()

##Merged Group
GroupbyCountry = dataset.groupby('Country').sum('USD (M)')
Mergedgroup=(pd.merge(GroupbyCountry, dataset2, on='Country'))
print(Mergedgroup)
Toploans =(Mergedgroup.sort_values(by=['USD (M)'], ascending=False))
print(Toploans)

##Top 5 by loans
Topfivebyloans=(Toploans.head(5))
print(Topfivebyloans)

width = 0.4
plt.bar(x=Topfivebyloans['Country'], height=Topfivebyloans['USD (M)'], color='blue', width=width, label='loans')
plt.ylabel('USD (M)', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.title('Loan Amount by Country', fontsize=12)
plt.legend()
plt.show()

plt.bar(x=Topfivebyloans['Country'], height=Topfivebyloans['Nominal GDP ($billions)'], color='Yellow', width=width, label='GDP')
plt.ylabel('USD (billions)', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.title('GDP by Country', fontsize=12)
plt.legend()
plt.show()


##Grouping
GroupbyCountry = dataset.groupby('Country').sum('USD (M)')
print(GroupbyCountry)

import matplotlib.pyplot as plt
GroupbySector=dataset.groupby('Sector').sum('USD (M)')
print(GroupbySector)
Sector = ['Agriculture','Banking', 'Budget', 'Business','Defense', 'Education','Environment','Government','Health','ICT','Industry','Mining','Multisector','Other social','Power','Transport','Unallocated','Water']
Amount = [288,2015,1650,815,1393,670,38,444,230,3661,1279,10133,2998,1069,18366,19219,642,3921]
plt.subplots()
plt.bar(x=Sector, height=Amount,color='Green',width=width, label='USD (M)');
plt.ylabel('Amount')
plt.xlabel('Sector')
plt.title('Sector by Amount')
plt.show()


## Visualisations (Seaborn scatterplot)
sns.set_style('darkgrid')
sns.relplot(data=dataset,x='Interest Rate', y='Country')
plt.title("Country vs Interest Rate")
plt.xlabel("Interest Rate")
plt.ylabel("Country")
plt.show()
















