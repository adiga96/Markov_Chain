import numpy as np
import pandas as pd
import random as rm
from scipy import stats

# Data Loading
data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')

## Functions

def increasing_rate_calculator(data_set):

    '''
    args:
        data_set (DataFrame) = Original COVID-19 data set
    returns:
        [Min, Avg, Max] rates (List) = Exponential rates of Increasing Hospitalization Trend
    '''

    data_copy = data_set # create a copy of the data
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
    inc_avg_rate = round(sum(increasing_rate_list) / INC_limit,5)


    #minimum and maximum rates observed in the in the given hospitalized data
    inc_min_rate = min(increasing_rate_list)
    inc_max_rate = max(increasing_rate_list)

    return [inc_min_rate, inc_avg_rate, inc_max_rate]

#-------------------------------------------------------------------------------------------------------------
def decreasing_rate_calculator(data_set):

    '''
    args:
        data_set (DataFrame) = Original COVID-19 data set
    returns:
        [Min, Avg, Max] rates (List) = Exponential rates of Decreasing Hospitalization Trend
    '''
    data_copy_2 = data_set
    INC_limit = 40
    hospital_list = data_copy_2['hospitalized'][INC_limit:].tolist() # decreasing trend

    def groupSequence(l): 
        '''
        args:
            l (List /array): List of values
        returns:
            increasing numbers groups [(1,2,3)(7,8,9)(21,24,27)] eg.
        '''
        start_bound = [i for i in range(len(l)-1) 
            if (l == 0 or l[i] != l[i-1]+1) 
            and l[i + 1] == l[i]+1] 
    
        end_bound = [i for i in range(1, len(l)) 
            if l[i] == l[i-1]+1 and
            (i == len(l)-1 or l[i + 1] != l[i]+1)] 
    
        return [l[start_bound[i]:end_bound[i]+1] 
        for i in range(len(start_bound))] 

    # Call the groupSequence function
    end_point_dec_trend = groupSequence(hospital_list)

    # To find the index of the value at which the rate data starts increasing
    index_end_point_dec_trend = hospital_list.index(end_point_dec_trend[0][0])
    Dec_list = data_copy_2[['hospitalized']][INC_limit : INC_limit + index_end_point_dec_trend + 1]  


    # create copy_2 of orignal data
    data_copy_3 = data_set
    dec = data_copy_3[['hospitalized']][INC_limit : INC_limit + index_end_point_dec_trend +1 ]
    Dec_list = dec.reset_index(drop=True)

    range_rate_list_1 = []
    DEC_limit = len(data_copy_3['hospitalized'][INC_limit: INC_limit + index_end_point_dec_trend + 1])
    
    i = 0
    for i in range(DEC_limit - 1):
        range_Rate_1 = math.log( (Dec_list['hospitalized'][i+1]) / (Dec_list['hospitalized'][i]))
        range_rate_list_1.append(range_Rate_1)

    dec_rate_list = [round(num,5) for num in range_rate_list_1] 
    dec_avg_rate = round(sum(dec_rate_list) / DEC_limit,5) # average decreasing exponential rate
    decreasing_rate_list  = dec_rate_list

    dec_min_rate = min(decreasing_rate_list)
    dec_max_rate = min(decreasing_rate_list, key=abs) #find value > 0 

    return [dec_min_rate, dec_avg_rate, dec_max_rate]

#------------------------------------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------------------------------------
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
                
    return accepted_value

# LikeLihood Ratio Test

'''
from scipy.stats.distributions import chi2
def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

LR = likelihood_ratio(L1,L2)
p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
'''

#-----------------------------------------------------------------------------------------------------------

# 2. Transition Probability Matrix Generator 

def Trans_Matrix(x,y):

    A = np.random.uniform(low = x, high = y)

    B_UL = 1 - A
    B_LL = 0
    B = np.random.uniform(low = B_LL , high = B_UL)

    C_UL = 1 - (A + B)
    C_LL = 0
    C = np.random.uniform(low = C_LL, high = C_UL)

    total_sum = A + B + C
    D = 1 - total_sum

    row_values = [A, B, C, D]

    return row_values

#--------------------------------------------------------------------------------------------------------------
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

#--------------------------------------------------------------------------------------------------------------
def posterior(obs_data,sim_data):
  
    
    accept = []
    for i in range(len(sim_data)):
        
        x = obs_data

        y = sim_data[i][0]
        y = obs_data
        K_S_Test = ks_2samp(x,y)
        p_value = K_S_Test.pvalue

        if(p_value >= 0.10):
            
            accept.append(sim_data[i])
    

    return accept

#--------------------------------------------------------------------------------------------------------------

# Testing MArkov Chain with Log Function

# NHC_increasing = [[value1, value2, value3, ....]
# initial values = [H, C, R, D]
# DTMC_predicted_values = []

def testing_DTMC_log_increasing(TP_Matrix,NHC_List):

    '''
    args:
        TP_Matrix (NumPy Matrix) = Transition Probability Matrix
        NHC_List (List) = Generated by Log Function

    returns:
        DTMC_predicted_values (NumPy array) = Predicted values, Transitions Probability Matrix

    '''

    DTMC_predicted_values = []
    
    DTMC_sub_array = [] # [Array[*Array*]]
    for i in range(4):
        DTMC_sub_array.append([])     #creating list inside list

    # print(type(NHC_List))
    # print(type(TP_Matrix))
    # print(len(NHC_List))

    for i in range(len(NHC_List)):
         
        current_state = np.matrix([NHC_List[i], 0, 0, 0]) #Taking values from the Log function NHC List
        # print(type(current_state))
        # print(type(TP_Matrix))
        new_state = current_state * TP_Matrix   #Markov Chain
        new_state = np.squeeze(np.asarray(new_state))   

        #append all values into a sub array
        DTMC_sub_array[0].append(new_state[0])
        DTMC_sub_array[1].append(new_state[1])
        DTMC_sub_array[2].append(new_state[2])
        DTMC_sub_array[3].append(new_state[3])
        
    # Subarrays and respective Transition Probabilities merged
    TP_Matrix = np.squeeze(np.asarray(TP_Matrix))
    DTMC_sub_array.append(TP_Matrix)

    # Merged Subarray to the main array (Accessed by Mainarray [i][j])
    DTMC_predicted_values.append(DTMC_sub_array)
        
    return DTMC_predicted_values

