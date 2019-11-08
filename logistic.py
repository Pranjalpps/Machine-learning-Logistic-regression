# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:43:48 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce


#importing the dataset
dataset = pd.read_csv('Iris.csv')
dataset=dataset[:100]
dataset2=dataset.copy()

encoder = ce.BinaryEncoder(cols=['Species'])
dataset2 = encoder.fit_transform(dataset2)


x=dataset2.iloc[:,1:3].values
y=dataset2.iloc[:,6].values


