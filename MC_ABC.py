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

# def TMatrix_generator():    
#     matrix = np.random.uniform(low=0., high=0.99, size=(4, 4))
#     matrix = matrix / matrix.sum(axis=1, keepdims=1)
#     # matrix = np.random.normal(size=(4, 4))
#     # matrix = matrix / matrix.sum(axis=1, keepdims=1)
#     new_matrix = matrix[:-2]
#     abs_states = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
#     trans_matrix = np.concatenate((new_matrix,abs_states))
#     TP_Matrix = np.matrix(trans_matrix)
#     return TP_Matrix

# 1. Logarthmic Function for Increasing and Decreasing Trends

def NHC(init_value,limit,rate):

    import math
    NHC_list = []
    present_state =init_value
    for i in range(limit):

        next_state = present_state * math.exp(rate)
        NHC_list.append(next_state)
        present_state = next_state
    
    return NHC_list

# 2. Transition Probability Matrix Generator (A)

def TMatrix_generator(x,y):    

    # matrix = np.random.uniform(low=0., high=0.99, size=(4, 4))
    matrix = np.random.uniform(low = x , high = y, size=(4,4))   
    matrix = matrix / matrix.sum(axis=1, keepdims=1)
    # matrix = np.random.normal(size=(4, 4))
    # matrix = matrix / matrix.sum(axis=1, keepdims=1)
    new_matrix = matrix[:-2]
    abs_states = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    trans_matrix = np.concatenate((new_matrix,abs_states))
    TP_Matrix = np.matrix(trans_matrix)
    return TP_Matrix

# 2. Transition Probability Matrix Generator (B)
'''
def TP(average_value):
    x = average_value + factor * average_value 
    y = average_value - factor * average_value
    gen = np.random.uniform(low = x, high = y)
    return gen
'''

def Trans_Matrix(x,y):
    A = np.random.uniform(low = x, high = y)
    B_UL = 1 - A
    B_LL = 0
    B = np.random.uniform(low = B_LL , high = B_UL)
    C_UL = 1 - B
    C_LL = 0
    C = np.random.uniform(low = C_LL, high = C_UL)
    total_sum = A + B + C
    D = 1- total_sum

    row_values = [A, B, C, D]

    return row_values

# 3. Simulation 

def simulationABC(initial_values, days, no_of_simulations, TP_mat):
    '''
    args:
        initial_values- The current state of hospitalized, critical care, recovered and death
        days = No. of days for which the simulation has to be run
        no_of_simulations - No of total times the the Transition Matrix has to be simulated    
    '''
    main_array = []
    

    for i in range(no_of_simulations):

        current_state = initial_values
        sub_array = [[i] for i in initial_values]
        TP_Matrix = TP_mat
        
        for j in range(days):
              
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


def posterior(obs_data,sim_data):
  
    
    accept = []
    for i in range(len(sim_data)):
        
        x = obs_data

        y = sim_data[i][0]
        y = obs_data
        K_S_Test = ks_2samp(x,y)
        p_value = K_S_Test.pvalue

        if(p_value >= 0.05):
            
            accept.append(sim_data[i])
    

    return accept
