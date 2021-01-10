import numpy as np
import pandas as pd
import math
from scipy import random
from scipy import stats
import statistics
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp  


data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')
# 1. Logarthmic Function for Increasing and Decreasing Trends

# Len(increasing_rate_list) = INC_Limit since we have deleted non-positive numbers and infinity from the list, total value will be equal to the increasing limit.

data_copy = data # create a copy of the data

# Find the Peak value of the distribution (MAX() or MIN())
INC_limit = data_copy.loc[data_copy['hospitalized'] == data_copy['hospitalized'].max()].index[0] 
# INC_limit = data_copy.loc[data_copy['hospitalized'] == data_copy['hospitalized'].max()].index[0] 


i = 0 #initialize to zero
range_rate_list = [] # list of the range values
for i in range(INC_limit):
    range_Rate = math.log((data_copy['hospitalized'][i+1]) / (data_copy['hospitalized'][i])) # ln(A/B) = X
    range_rate_list.append(range_Rate)

inc_rate_list = [round(num,5) for num in range_rate_list] #list of rnage values rounded to 5
num_list = []
for num in inc_rate_list:   
    # checking condition 
    if num >= 0: 
        num_list.append(num)
        
increasing_rate_list = num_list[1:] #To remove the 1st element since it was Infinity 

# Average rate observed in the given hospitalized data
increasing_exp_rate = round(sum(increasing_rate_list) / INC_limit,5)

#minimum and maximum rates observed in the in the given hospitalized data
min_rate = min(increasing_rate_list)
max_rate = max(increasing_rate_list)

# Fucntion to get the NHC change rate using simulation
def exp_simulation(max_rate, min_rate, total_simulations, init_value, total_values):
    '''
    args:
        #  For simulation = max_rate, min_rate
        1. increasing_exp_rate (float) = Average Rate of New Hospitalization Cases increasing from the OBSERVED DATA
        2. total_simulations (integer) = Total Number of simulations that has to be conducted
        3. init_value (integer) = Initial Value for the LOG Function
        4. total_values (integer) = Total Values to be calculated using the LOG Function
        5. observed_values (List) = Hospitalizations Cases from OBSERVED DATA
    
    returns:
        array[[array][array]] with all values simulated from the NHC_rate generated 

    '''
    # NHC = New Hospitalization Cases

    NHC_list = [] 

    for i in range(total_simulations):

        rate = np.random.uniform(low = min_rate, high = max_rate) # For simulation
        #rate = 0.1015 # Average Rate from Empirical Data
        limit = total_values # Total values to predicted using LOG Function
        present_state = init_value # Initial Value to start the prediction with

        NHC_sub_list = [] # [Array[*Array*]]
        for i in range(2): 
            NHC_sub_list.append([])     #creating list inside list
        
        for j in range(limit):

            next_state = present_state * math.exp(rate) # NHC(n+1) = NHC(n) x e ^ (NHC_rate)
            NHC_sub_list[0].append(next_state)
            present_state = next_state

        NHC_sub_list[1].append(rate)
        NHC_list.append(NHC_sub_list) #appending the values of each simulation to main list
    

    return NHC_list


# Kolmogorov Smirnov Test
def KS_TEST(NHC_list, observed_values):

    '''
    args:
        NHC_list (List[a][b]) = Output of exp_simulation Function
        observed_values = OBSERVED increasing trend data

    returns: 
        All the accepted values by checking p value and matching significance 
    '''
    accepted_values = []

    for i in range(len(NHC_list)):

        K_S_Test = ks_2samp(NHC_list[i][0],observed_values)
        p_value = K_S_Test.pvalue

        if p_value >= 0.10: 
            accepted_values.append(NHC_list[i])
            # print('Values accepted')
        # else:
            # print('Values rejected')
                
    return accepted_values

# LikeLihood Ratio Test

from scipy.stats.distributions import chi2
def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

LR = likelihood_ratio(L1,L2)
p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1



# Calculating Likelihood of Curve Fitting using Scipy
# https://stackoverflow.com/questions/23004374/how-to-calculate-the-likelihood-of-curve-fitting-in-scipy

import scipy.optimize as so
import scipy.stats as ss
# xdata = np.array([-2,-1.64,-1.33,-0.7,0,0.45,1.2,1.64,2.32,2.9])
# ydata = np.array([0.699369,0.700462,0.695354,1.03905,1.97389,2.41143,1.91091,0.919576,-0.730975,-1.42001])

xdata = np.array(q2)
ydata = np.array(q1)

def model0(x, p1, p2):
    return p1 * np.cos(p2 * x) + p2 * np.sin(p1 * x)
def model1(x, p1, p2, p3):
    return p1 * np.cos(p2 * x) + p2 * np.sin(p1 * x) + p3 * x

p1, p2, p3 = 1, 0.2, 0.01

fit0 = so.curve_fit(model0, xdata, ydata, p0=(p1, p2))[0]
fit1 = so.curve_fit(model1, xdata, ydata, p0=(p1, p2, p3))[0]
yfit0 = model0(xdata, fit0[0], fit0[1])
yfit1 = model1(xdata, fit1[0], fit1[1], fit1[2])

ssq0 = ((yfit0-ydata)* 2).sum()
ssq1 = ((yfit1 - ydata)* 2).sum()

df = len(xdata) - 3

f_ratio = (ssq0 - ssq1) / (ssq1 / df)
p = 1 - ss.f.cdf(f_ratio, 1, df)

# Decreasing Function

data_copy_2 = data

dec = data_copy_2[['hospitalized']][INC_limit : INC_limit + index_end_point_dec_trend + 1 ]
Dec_list = dec.reset_index(drop=True)

range_rate_list_1 = []

DEC_limit = len(data_copy_2['hospitalized'][INC_limit: INC_limit + index_end_point_dec_trend + 1])

i = 0
for i in range(DEC_limit - 1):
    range_Rate_1 = math.log( (Dec_list['hospitalized'][i+1]) / (Dec_list['hospitalized'][i]))
    range_rate_list_1.append(range_Rate_1)

dec_rate_list = [round(num,5) for num in range_rate_list_1] 
decreasing_exp_rate = round(sum(dec_rate_list) / DEC_limit,5)

# To identify the value at which the rate data starts increasing 

def groupSequence(l): 
    start_bound = [i for i in range(len(l)-1) 
        if (l == 0 or l[i] != l[i-1]+1) 
        and l[i + 1] == l[i]+1] 
  
    end_bound = [i for i in range(1, len(l)) 
        if l[i] == l[i-1]+1 and
        (i == len(l)-1 or l[i + 1] != l[i]+1)] 
  
    return [l[start_bound[i]:end_bound[i]+1] 
    for i in range(len(start_bound))] 
    
    end_point_dec_trend = groupSequence(hospital_list)

# To find the index of the value at which the rate data starts increasing

index_end_point_dec_trend = hospital_list.index(end_point_dec_trend[0][0])
Dec_list = data_copy_2[['hospitalized']][INC_limit : 40 + index_end_point_dec_trend + 1]
