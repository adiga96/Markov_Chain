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

def TMatrix_generator():    
    matrix = np.random.uniform(low=0., high=0.99, size=(4, 4))
    matrix = matrix / matrix.sum(axis=1, keepdims=1)
    # matrix = np.random.normal(size=(4, 4))
    # matrix = matrix / matrix.sum(axis=1, keepdims=1)
    new_matrix = matrix[:-2]
    abs_states = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    trans_matrix = np.concatenate((new_matrix,abs_states))
    TP_Matrix = np.matrix(trans_matrix)
    return TP_Matrix

# import pdb
def TMatrix_generator():    

    matrix = np.random.uniform(low=0., high=0.99, size=(4, 4))
    matrix = matrix / matrix.sum(axis=1, keepdims=1)
    # matrix = np.random.normal(size=(4, 4))
    # matrix = matrix / matrix.sum(axis=1, keepdims=1)
    new_matrix = matrix[:-2]
    abs_states = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    trans_matrix = np.concatenate((new_matrix,abs_states))
    TP_Matrix = np.matrix(trans_matrix)
    return TP_Matrix
    
def simulationABC(initial_values, days, no_of_simulations):

    main_array = []

    for i in range(no_of_simulations):

        sub_array = [[i] for i in initial_values]

        for j in range(days):
            
            current_state = initial_values
            TP_Matrix = TMatrix_generator()
            new_state = current_state * TP_Matrix
            new_state = np.squeeze(np.asarray(new_state))

            sub_array[0].append(new_state[0])
            sub_array[1].append(new_state[1])
            sub_array[2].append(new_state[2])
            sub_array[3].append(new_state[3])

            current_state = new_state       

        TP_Matrix = np.squeeze(np.asarray(TP_Matrix))
        sub_array.append(TP_Matrix)

        main_array.append(sub_array)


    return main_array
  
import matplotlib.pyplot as plt
plt.plot(sim_1['H'],color='red',label='Hospitalized')
plt.plot(sim_1['C'],color='green',label='Critical Care')
plt.plot(sim_1['R'],color='yellow',label='Recovered')
plt.plot(sim_1['D'],color='blue',label='Death')