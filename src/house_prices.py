from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv("tests/data/train.csv", index_col='Id')
X_test_full = pd.read_csv("tests/data/test.csv", index_col='Id')
print("Old Shape: ", X_full.shape)
# Remove outliers
Q1 = np.percentile(X_full.SalePrice, 25, method='midpoint')
Q3 = np.percentile(X_full.SalePrice, 75, method='midpoint')
IQR = Q3 - Q1
upper = Q3+1.5*IQR
lower = Q1-1.5*IQR
# Create arrays of Boolean values indicating the outlier rows
upper_array = np.where(X_full.SalePrice >= upper)[0]
lower_array = np.where(X_full.SalePrice <= lower)[0]

print("Upper Bound:", upper)
print(upper_array.sum())

print("Lower Bound:", lower)
print(lower_array.sum())
outliers_indexes = np.concatenate((upper_array, lower_array))
print(outliers_indexes)
X_no_outliers = X_full.drop(index=outliers_indexes)

print("New Shape: ", X_no_outliers.shape)

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF',
            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.80, test_size=0.2,
                                                      random_state=0)

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(
    n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(
    n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]


# Function for comparing different models

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)


maes = []
for model in models:
    maes.append(score_model(model))
    print(f"Model {models.index(model)+1} MAE: {score_model(model)}")


best_model_index = maes.index(min(maes))
# Fill in the best model
best_model = models[best_model_index]

# Define a model
my_model = best_model

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
