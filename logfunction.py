import numpy as np
import pandas as pd
import math
from scipy import random
from scipy import stats
import statistics
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp  


data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')

data_copy = data
INC_limit = data_copy.loc[data_copy['hospitalized'] == data_copy['hospitalized'].max()].index[0]

i = 0
range_rate_list = []
for i in range(INC_limit):
    range_Rate = math.log((data_copy['hospitalized'][i+1]) / (data_copy['hospitalized'][i]))
    range_rate_list.append(range_Rate)

inc_rate_list = [round(num,5) for num in range_rate_list]
num_list = []
for num in inc_rate_list:   
    # checking condition 
    if num >= 0: 
        num_list.append(num)
        
increasing_rate_list = num_list[1:]

# Len(increasing_rate_list) = INC_Limit since we have deleted non-positive numbers and infinity from the list, total value will be equal to the increasing limit.
increasing_exp_rate = round(sum(increasing_rate_list) / INC_limit,5)

def exp_simulation(rate, total_simulations, init_value, total_values):
    
    NHC_list = [] 
    
    for i in range(total_simulations):

        rate = np.random.uniform(low = rate - 0.025 , high = rate + 0.025)
        limit = total_values
        present_state = init_value

        NHC_sub_list = []
        for i in range(2): 
            NHC_sub_list.append([])     #creating list inside list
        
        for j in range(limit):

            next_state = present_state * math.exp(rate)
            NHC_sub_list[0].append(next_state)
            present_state = next_state

        NHC_sub_list[1].append(rate)
        NHC_list.append(NHC_sub_list) #appending the values of each simulation to main list

    return NHC_list
