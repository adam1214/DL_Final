# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:05:10 2021

@author: Admin
"""

import pickle

new_all_data_path = './new_all_data.pickle'
f = open(new_all_data_path, 'rb') 
new_all_data_dict = pickle.load(f)