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
def MC_calculator(transition_matrix,present_state, days):

    # present_df = pd.DataFrame([present_state], columns=list('HCRD'))
  
    array = [[i] for i in present_state]
    # print(transition_matrix)
    
    for j in range(days):
        new_state = present_state * transition_matrix
        new_state = np.squeeze(np.asarray(new_state))

        array[0].append(new_state[0])
        array[1].append(new_state[1])
        array[2].append(new_state[2])
        array[3].append(new_state[3])
        
        present_state = new_state
        
    # for i in range(days):
    #     new_state = present_state * transition_matrix
    #     new_state_1 = np.squeeze(np.asarray(new_state))
    #     for i in new_state_1:
    #          somelist.append([i])

    #     # new_df = pd.DataFrame([new_state_1], columns=list('HCRD'))
    #     # tmatrix_df = pd.DataFrame([t_matrix], columns=list('T_Matrix'))
    #     # tmatrix_df = pd.DataFrame(data=transition_matrix.flatten(), columns=["Transition Matrix"])
    #     # present_df = present_df.append(new_df,ignore_index=True)
    #     # present_df = present_df.append(tmatrix_df,ignore_index=True)
    
    transition_matrix = np.squeeze(np.asarray(transition_matrix))
    array.append(transition_matrix)

    return array
  
import matplotlib.pyplot as plt
plt.plot(sim_1['H'],color='red',label='Hospitalized')
plt.plot(sim_1['C'],color='green',label='Critical Care')
plt.plot(sim_1['R'],color='yellow',label='Recovered')
plt.plot(sim_1['D'],color='blue',label='Death')
