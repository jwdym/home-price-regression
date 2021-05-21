# Import modules
import os
import pickle
import json
import sklearn

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

base_path = os.getcwd()

# Load encoder
print('Loading encoder')
filename = 'encoder.pkl'
with open(f'model/{filename}', 'rb') as file:
    enc = pickle.load(file)

# Load columns
print('Loading columns')
filename = 'columns.json'
with open(f'model/{filename}') as json_file:
    cols = json.load(json_file)

cat_cols = cols['cat_cols']
num_cols = cols['num_cols']
na_cols = cols['na_cols']

# Load imputer
print('Loading imputer')
filename = 'imputer.pkl'
with open(f'model/{filename}', 'rb') as file:
    imputer = pickle.load(file)

# Load model
print('Loading model')
filename = 'model.pkl'
with open(f'model/{filename}', 'rb') as file:
    model = pickle.load(file)

# Load input data
print('Loading data')
test_dat = pd.read_csv(f'{base_path}/data/test.csv')

# Transform variables
print('Transforming data')
test_dat['Remodeled'] = (test_dat.YearRemodAdd != test_dat.YearBuilt) * 1
test_dat['Years_Since_Remodel'] = test_dat.YearRemodAdd - test_dat.YearBuilt
test_dat['Home_Age'] = test_dat.YrSold - test_dat.YearBuilt
test_dat['cat_YrSold'] = test_dat.YrSold.astype(str)
test_dat['cat_MoSold'] = test_dat.MoSold.astype(str)

enc.transform(test_dat[cat_cols]).toarray()
test_imp = pd.concat([pd.DataFrame(enc.transform(test_dat[cat_cols]).toarray()), test_dat[num_cols], test_dat[na_cols].isna() * 1], axis=1)
x_test = imputer.transform(test_imp)

# Save output
print('Saving model output')
output = pd.DataFrame({'Id': test_dat.Id, 'SalePrice': model.predict(x_test)})
output.to_csv(f'{base_path}/data/jdym_submission.csv', index=False)