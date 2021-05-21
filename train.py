# Import modules
import os
import pickle
import sklearn
import json

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import r2_score, explained_variance_score, max_error

# Load data
print('Loading data')
base_path = os.getcwd()
dat = pd.read_csv(f'{base_path}/data/train.csv')

# Remove Id column
print('Transforming data')
dat = dat.drop('Id', axis=1)

# Create new features 
dat['Remodeled'] = (dat.YearRemodAdd != dat.YearBuilt) * 1
dat['Years_Since_Remodel'] = dat.YearRemodAdd - dat.YearBuilt
dat['Home_Age'] = dat.YrSold - dat.YearBuilt

# Identify numerical data type columns
num_cols = list(dat.columns[dat.dtypes == np.int64])
num_cols += list(dat.columns[dat.dtypes == np.float64])

# Identify categorical data type columns
cat_cols = list(set(dat.columns) - set(num_cols))

# Drop columns with lots of missing data and target column
drops = ['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature', 'SalePrice']
na_cols = ['Alley', 'PoolQC', 'Fence', 'FireplaceQu', 'MiscFeature']
cat_cols = list(set(cat_cols) - set(drops))
num_cols = list(set(num_cols) - set(drops))

# Convert date columns to categorical
dat['cat_YrSold'] = dat.YrSold.astype(str)
dat['cat_MoSold'] = dat.MoSold.astype(str)
cat_cols += ['cat_YrSold', 'cat_MoSold']

# Save columns
print('Saving column categories')
cols = {'na_cols': na_cols, 'cat_cols': cat_cols, 'num_cols': num_cols}
filename = 'columns.json'
with open(f'model/{filename}', 'w') as f:
    json.dump(cols, f)

# Setup categorical data encoder
print('Encoding categorical data')
cat_categories = []
for c in cat_cols:
    cat_categories.append(list(set(dat[c])))

enc = OneHotEncoder(categories=cat_categories, handle_unknown='ignore')
enc.fit(dat[cat_cols])

# Save encoder
print('Saving one hot encoder')
filename = 'encoder.pkl'
with open(f'model/{filename}', 'wb') as file:
    pickle.dump(enc, file)

# Format training data and impute missing values
print('Imputing missing values')
dat_imp = pd.concat([pd.DataFrame(enc.transform(dat[cat_cols]).toarray()), dat[num_cols], dat[na_cols].isna() * 1], axis=1)
imputer = KNNImputer(n_neighbors=5)
x_train = imputer.fit_transform(dat_imp)
y_train = dat.SalePrice

# Save imputer
print('Saving missing data imputer')
filename = 'imputer.pkl'
with open(f'model/{filename}', 'wb') as file:
    pickle.dump(imputer, file)

# Run hyperparameter job to get optimal model
print('Running hyperparameter tuning job')
param_grid = {
    'loss': ('ls', 'huber', 'quantile'),
    'learning_rate': list(np.linspace(0.01, 0.3, 20)),
    'n_estimators': list(np.linspace(100, 300, 20, dtype=int)),
    'criterion': ('friedman_mse', 'mse'),
    'subsample': list(np.linspace(0.4, 1, 20)),
    'max_depth': list(np.linspace(1, 12, 12, dtype=int))
    }

base_estimator = GradientBoostingRegressor()
sh = HalvingRandomSearchCV(estimator=base_estimator, param_distributions=param_grid, scoring='explained_variance', cv=4, factor=3, max_resources='auto', aggressive_elimination=False).fit(x_train, y_train)
hp_df = pd.DataFrame(sh.cv_results_)

# Get model
# HP job parameters
params = hp_df.loc[hp_df.rank_test_score == 1].params.values[0]

# HP job model
model = sh.best_estimator_

# Save model
filename = 'model.pkl'
with open(f'model/{filename}', 'wb') as file:
    pickle.dump(model, file)

# Model evaluation metrics
print(f'Training Data Correlation: {r2_score(y_train, model.predict(x_train))}')
print(f'Training Data Explained Variance: {explained_variance_score(y_train, model.predict(x_train))}')
print(f'Training Data Max Error: {max_error(y_train, model.predict(x_train))}')