# Convert housing.csv into a dataframe
import pandas as pd
import numpy as np
#import matplotlib.pyplotnnumpy

housing = pd.read_csv("housing.csv")
#Impute missing values with median
#housing['total_bedrooms'] = housing['total_bedrooms'].fillna(housing['total_bedrooms'].median())
#print(housing.info())
# Split the data into training and test data sets
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
# Create a copy of the training set to play around with
housing = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()
#Add additional attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
# Collect the columns containing the numerical attributes
housing_num = housing.drop("ocean_proximity", axis=1)
print(housing_num.info())
#convert categorical attribute to numerical
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))
#Convert categorical attributes to one-hot vectors
num_attribs = list(housing_num)
cat_attribs = list(housing_cat)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scaler', StandardScaler()),])
from sklearn.compose import ColumnTransformer
full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),("cat", OneHotEncoder(), cat_attribs)])
housing_prepared = full_pipeline.fit_transform(housing)
print(len(housing_prepared))
#Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
#Try it out on some data
some_data = housing.iloc[1000:1005]
some_labels = housing_labels.iloc[1000:1005]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_predictions, housing_labels)
lin_rmse = np.sqrt(lin_mse)
print(lin_mse)
# Train a decision tree regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)
# Perform cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Dev:", scores.std())
display_scores(tree_rmse_scores)
#Random forest regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
# Moving on to testing the Random Forest regressor on the test set
#Prepare the test set
X_test = test_set.drop("median_house_value", axis=1)
Y_test = test_set["median_house_value"].copy()
X_test["rooms_per_household"] = X_test["total_rooms"]/X_test["households"]
X_test["bedrooms_per_room"] = X_test["total_bedrooms"]/X_test["total_rooms"]
X_test["population_per_household"] = X_test["population"]/X_test["households"]
X_test_prepared = full_pipeline.fit_transform(X_test)
final_predictions = forest_reg.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
#
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(cat_encoder.categories_)
# housing_cat_array = housing_cat_1hot.toarray()
# housing_cat_num = pd.DataFrame(housing_cat_array)
# #Concatenate the numerical and categorical data frames
# housing_prepared = pd.concat([housing_num, housing_cat_num], axis=1)
# print(housing_prepared.head(10))
# # Plot the histogram
# housing.hist(bins=50, figsize=(20, 15))
# housing.hist(column='median_income', bins=50, figsize=(20, 15))
# plt.show()

# print(housing.info())
# # Correlation plots
# housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c=housing["median_house_value"], cmap=plt.get_cmap("jet"), colorbar=True,)
# plt.show()
# # Adding new attributes
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
# # Look at the correlation matrix
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))
#


