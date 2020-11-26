import numpy as np
import pandas as pd
import random as rm
from scipy import stats

# Read the data
data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')
# data.columns

# append to list
hospitalized = data['hospitalized'].tolist()
critical = data['Critical'].tolist()
recovered = data['recovered'].tolist()
death = data['death'].tolist()

# summary statisitics of given data
stats.describe(hospitalized)   # as required

from fitter import Fitter
f = Fitter(hospitalized)
f.fit()
# may take some time since by default, all distributions are tried
# but you call manually provide a smaller set of distributions
f.summary()

N = 1000000
for i in range(N):    
    matrix = np.random.uniform(low=0., high=0.99, size=(4, 4))
    matrix = matrix / matrix.sum(axis=1, keepdims=1)

    new_matrix = matrix[:-2]
    abs_states = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    trans_matrix = np.concatenate((new_matrix,abs_states))
    TP_Matrix = np.matrix(trans_matrix)
    print(TP_Matrix)
