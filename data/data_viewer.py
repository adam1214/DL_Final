import pandas as pd
import pdb
from collections import Counter
import numpy as np
import math
'''
train_X = []
train_Y = []
train_data = pd.read_csv('./train.csv', error_bad_lines=False)
test_data = pd.read_csv('./test.csv', error_bad_lines=False)
for index, row in train_data.iterrows():
    train_X.append(row[0:-1])
    train_Y.append(row[-1])

train_X = np.asarray(train_X)
for i in range(0, train_X.shape[0], 1):
    for j in range(0, train_X.shape[1], 1):
        try:
            if math.isnan(train_X[i][j]):
                train_X[i][j] = -1.0
        except:
            continue
            
train_Y = np.asarray(train_Y)
counter = Counter(train_Y)
print(counter)
'''

country_dict = {}
location_dict = {}
station_dict = {}

train_data = pd.read_csv('./train.csv', error_bad_lines=False)
train_data_copy = train_data.copy()

for index, row in train_data.iterrows():
    if station_dict.get(row[0]) != None:
        row[0] = station_dict[row[0]]
    else:
        station_num = len(station_dict)
        station_dict[row[0]] = station_num
        row[0] = station_dict[row[0]]
        
    if country_dict.get(row[2]) != None:
        row[2] = country_dict[row[2]]
    else:
        country_num = len(country_dict)
        country_dict[row[2]] = country_num
        row[2] = country_dict[row[2]]
    
    if location_dict.get(row[3]) != None:
        row[3] = location_dict[row[3]]
    else:
        location_num = len(location_dict)
        location_dict[row[3]] = location_num
        row[3] = location_dict[row[3]]
    
    train_data_copy.loc[index] = row
    for j in range(0, len(row), 1):
        try:
            if math.isnan(row[j]):
                row[j] = -1.0
                train_data_copy.loc[index] = row
        except:
            continue
            
        