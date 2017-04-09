import pandas as pd
import numpy as np

col_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
             'PTRATIO', 'B', 'LSTAT', 'MEDV']

df = pd.read_csv('housing_train.txt', sep='\t', delim_whitespace=True, header=None, names=col_names)

print(df)
