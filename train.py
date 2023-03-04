from sklearn.linear_model import LogisticRegression
import pandas as pd
from utils import split_data
import numpy as np

data = pd.read_excel('clean_data.xlsx').to_numpy()[...,1:]
train_data,train_labels,test_data,test_labels = split_data(data)

model = LogisticRegression(random_state=0,max_iter = 1000).fit(train_data, train_labels)

print(model.score(test_data, test_labels))
