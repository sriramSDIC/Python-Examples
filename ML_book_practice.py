#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:09:24 2020

@author: snarasim
"""
# Convert housing.csv into a dataframe
import pandas as pd
import matplotlib.pyplot as plt

housing = pd.read_csv("housing.csv")
print(housing.head())

# Plot the histogram
housing.hist(bins=50, figsize=(20, 15))
housing.hist(column='median_income', bins=50, figsize=(20, 15))
plt.show()
# Split the data into training and test data sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Create a copy of the training set to play around with
housing = train_set.copy()
# Correlation plots
housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c=housing["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True,)
plt.show()
# Adding new attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# Look at the correlation matrix
corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


