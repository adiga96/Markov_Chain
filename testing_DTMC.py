%load_ext autotime
import numpy as np
import pandas as pd
import math
from scipy import random
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import pickle

# Data Type 1
data_1 = pd.read_csv('//Users/pritishsadiga/Desktop/MS_Thesis/Data_Testing/data_test.csv')
# Data Type 2
data = pd.read_csv('/Users/pritishsadiga/Desktop/MS_Thesis/Data_Testing/data.csv')

observed_values = [data['hospitalized'].tolist(),
                    data['Critical'].tolist(),
                    data['recovered'].tolist(),
                    data['death'].tolist()]

observed_values_1 = [data_1['hospitalizedCurrently'].tolist(),
                    data_1['CriticalCurrently'].tolist(),
                    data_1['RecoveredCurrently'].tolist(),
                    data_1['DeathCurrently'].tolist()]

# To calculate the rate of increase of cases from March 1 to March 17
data_copy = data
data_filter = data_copy[['hospitalized']][15:] #since 0-13 all values are zero
data_filter = data_filter.reset_index(drop = True)
mylist = data_filter['hospitalized'].tolist()

new_hospitalized = mylist
# yo

days = 17
NHC_cases = []
rate = 0.3405 # first 17 days no data give, so assume a rate 34.05%
current_value = 1 # assume on March 1, hospitalization cases = 1
for i in range(days):
    next_value = current_value * math.exp(rate)
    NHC_cases.append(next_value)
    current_value = next_value

# NHC_cases = [round(num) for num in NHC_cases]
# NHC_cases

# To calculate the 7 days, 1 week rate of addition of NHC (New Hopsitalization Cases)
rate = 0
rate_count = 0
NHC_avg_rate = [] 

for i in range(0, len(new_hospitalized) - 1):

    rate += math.log(new_hospitalized[i + 1] / new_hospitalized[i])
    rate_count += 1        #count every iteration

    if rate_count == 7:     #once count is equal to a week, caluclate average rate
        NHC_avg_rate.append(rate / 7)
        rate = 0
        rate_count = 0

# To generate NHC based on 1-week rate change
count = 0
NHC_cases_1 = []
pp = NHC_cases[16]
# pp

for i in range(0,len(new_hospitalized)-1):
    # print(i)
    rate = NHC_avg_rate[count // 7]
    # print(rate)
    nn = pp * math.exp(rate)
    NHC_cases_1.append(nn)
    pp = nn

    count+=1
    if count // 7 >= len(NHC_avg_rate):
        break

NHC_total_cases = NHC_cases + NHC_cases_1
NHC_total_cases = [round(num,2) for num in NHC_total_cases]
# NHC_total_cases

def Trans_Matrix_II(total_values):
  
    P_HH = (np.random.uniform(0.65, 0.85, total_values)).round(4)
    P_CH = (np.random.uniform(0.0, 0.25, total_values)).round(4)
    P_RH = (np.random.uniform(0.0, 0.1, total_values)).round(4)
    P_DH = (np.random.uniform(0.0, 0.1, total_values)).round(4)

    P_HC = (np.random.uniform(0.35, 0.6, total_values)).round(4)
    P_CC = (np.random.uniform(0.30, 0.6, total_values)).round(4)
    P_RC = (np.random.uniform(0.0, 0.15, total_values)).round(4)
    P_DC = (np.random.uniform(0.0, 0.15, total_values)).round(4)

    return [    P_HH, P_CH, P_RH, P_DH, 
                P_HC,P_CC,P_RC,P_DC    ]
   
   # 1000 Prior Probabilities
prior_prob_1K = Trans_Matrix_II(1000)
# 10000 Prior Probabilities
prior_prob_10K = Trans_Matrix_II(10000)
# 100000 Prior Probabilities
prior_prob_100K = Trans_Matrix_II(100000)


# data = np.random.uniform(0.10,0.20,1000) # You are generating 1000 points between 0 and 1.
def PLOT_PRIOR(prior, title):

    count, bins, ignored = plt.hist(prior, 35, facecolor='green') 

    plt.xlabel('Probabilities')
    plt.ylabel('Count')
    plt.title(title)
    # plt.axis([min_prior, 1, 0, y_end]) # x_start, x_end, y_start, y_end
    plt.grid(True)
    return plt.show(block = False)
   
  def PlOT_PRIOR_SIM(prior):  
    HH_prior = PLOT_PRIOR(prior[0],'Prior: P(H|H)')
    CH_prior = PLOT_PRIOR(prior[1],'Prior: P(C|H)')
    RH_prior = PLOT_PRIOR(prior[2],'Prior: P(R|H)')
    DH_prior = PLOT_PRIOR(prior[3],'Prior: P(D|H)')

    HC_prior = PLOT_PRIOR(prior[4],'Prior: P(H|C)')
    CC_prior = PLOT_PRIOR(prior[5],'Prior: P(C|C)')
    RC_prior = PLOT_PRIOR(prior[6],'Prior: P(R|C)')
    DC_prior = PLOT_PRIOR(prior[7],'Prior: P(D|C)')

    return HH_prior, CH_prior, RH_prior, DH_prior, HC_prior, CC_prior, RC_prior, DC_prior
   
  # prior_plots_1K = PlOT_PRIOR_SIM(prior_prob_1K)
  # prior_plots_10K = PlOT_PRIOR_SIM(prior_prob_10K)
  # prior_plots_100K = PlOT_PRIOR_SIM(prior_prob_100K)
  
  def TP_Matrix_row(P_HH, P_CH, P_RH, P_DH):

    '''
        args:
            P_HH, P_CH, P_RH, P_DH (array of uniform distribution)
        returns:
            row (array) = [[converted values], [original values]]
    '''
    
    first_row = []
    for i in range(len(P_HH)):
        row_sum = P_HH[i] + P_CH[i] + P_RH[i] + P_DH[i]
        row_11 = P_HH[i] / row_sum
        row_12 = P_CH[i] / row_sum
        row_13 = P_RH[i] / row_sum
        row_14 = P_DH[i] / row_sum
        first_row_list = [[row_11, row_12, row_13, row_14],row_sum]
        first_row.append(first_row_list)
    
    return first_row
   
   def STOCH_Matrix_original_sum(prior_simulations, total_sims):

    '''
        args:
            prior_simulations (array) : simulated prior parameters
        returns:
            total_trans_matrix_list (array) : A list containing STOCHASTIC MATRIX and ORIGINAL SUM OF VALUES

    '''
    
    first_row_values = TP_Matrix_row(prior_simulations[0], prior_simulations[1], prior_simulations[2], prior_simulations[3])
    second_row_values = TP_Matrix_row(prior_simulations[4], prior_simulations[5], prior_simulations[6], prior_simulations[7])
    third_row_values = [0, 0, 1, 0]
    fourth_row_values = [0, 0, 0, 1]

    total_trans_matrix_list = []
    for i in range(total_sims):

            # Transition Matrix
            matrix = np.matrix([first_row_values[i][0], second_row_values[i][0], third_row_values, fourth_row_values])

            # Sum of Original Values of each row
            original_row_sum_values = [first_row_values[i][1], second_row_values[i][1], 1, 1]

            # Final list containing STOCHASTIC TRANSITION MATRIX and the ORIGINAL SUM VALUES 
            final_list = [matrix, original_row_sum_values]
            
            total_trans_matrix_list.append(final_list)

    return total_trans_matrix_list
   
   
   Stoc_matrix_sum_1K = STOCH_Matrix_original_sum(prior_prob_1K, 1000)
   Stoc_matrix_sum_10K = STOCH_Matrix_original_sum(prior_prob_10K, 10000)
   Stoc_matrix_sum_100K = STOCH_Matrix_original_sum(prior_prob_100K, 100000)
   
   # Function to write files into TXT

def file_txt_store(file_list, sim_name, date):   
    with open('/Users/pritishsadiga/Desktop/MS_Thesis/Method2_results/prior'+ str(sim_name) + str(date) + '.txt' , 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(file_list, filehandle)
        return print('File_stored')


     # To store the prior parameters of the transition probability matrix 
file_txt_store(Stoc_matrix_sum_1K, 'TMatrix1K', '02_22_24') 
file_txt_store(Stoc_matrix_sum_10K, 'TMatrix10K', '02_22_24')
file_txt_store(Stoc_matrix_sum_100K, 'TMatrix100K', '02_22_24')
# file_txt_store(Stoc_matrix_sum_1000K, 'TMatrix1000K', '02_21_21')

def testing_DTMC_M2(Transition_matrix, original_sum, NHC_List):
 
    '''
    args:
    NHC_List (List) = Generated by Log Function
    Total_simulations = No of simulations to be conducted

    returns:
    DTMC_predicted_values (NumPy array) = Predicted values, Transitions Probability Matrix

    '''
    DTMC_predicted_values = [[],[]]

    DTMC_sub_array = [] # [Array[*Array*]]
    DTMC_sub_array_2 = []

    for i in range(4):
        DTMC_sub_array.append([])     #creating list inside list

    value = NHC_List[0]
    current_state = np.matrix([value, 0, 0, 0])
    current_state = np.squeeze(np.asarray(current_state))   

    DTMC_sub_array[0].append(current_state[0])
    DTMC_sub_array[1].append(current_state[1])
    DTMC_sub_array[2].append(current_state[2])
    DTMC_sub_array[3].append(current_state[3])

    # iteration 1
    n_state = current_state * Transition_matrix
    n_state = np.squeeze(np.asarray(n_state)) 

    DTMC_sub_array[0].append(NHC_List[1])
    DTMC_sub_array[1].append(n_state[1])
    DTMC_sub_array[2].append(n_state[2])
    DTMC_sub_array[3].append(n_state[3])

    for i in range(1, len(NHC_List)):

        current_state = np.matrix([NHC_List[i], DTMC_sub_array[1][i-1], DTMC_sub_array[2][i-1], DTMC_sub_array[3][i-1]])
        new_state = current_state * Transition_matrix #Markov Chain
        new_state = np.squeeze(np.asarray(new_state)) 

        DTMC_sub_array[0].append(NHC_List[i])
        DTMC_sub_array[1].append(new_state[1])
        DTMC_sub_array[2].append(new_state[2])
        DTMC_sub_array[3].append(new_state[3])

    # Subarrays and respective Transition Probabilities merged
    Transition_matrix = np.squeeze(np.asarray(Transition_matrix))
    DTMC_sub_array.append(Transition_matrix)
    DTMC_sub_array_2.append(original_sum)

    DTMC_predicted_values[0].append(DTMC_sub_array)
    DTMC_predicted_values[1].append(DTMC_sub_array_2)

    return DTMC_predicted_values
   
   def simulate_DTMC_M2(Trans_prob__matrix, NHC_total_cases):
    '''
    args:
        NHC_total_cases (List)  = Total New Hospitalization Cases 
        total_sims (int) = Total number of simulations to be run

    returns:
        row1 = First row of Tranisiton Probability Matrix
        row2 = Second row of Transition Probability Matrix
        simulated_values = Output of H, C, R, D
    '''
    
    simulated_values = []

    # A1 = np.random.uniform(0.70,0.90,total_sims)
    # A2 = np.random.uniform(0.30,0.70,total_sims)

    # row1 = Trans_Matrix(A1)
    # row2 = Trans_Matrix(A2)

        # load additional module

    for sim in range(0, len(Trans_prob__matrix)):

        # print(sim)
        # print(len(Trans_prob__matrix))
        TP_matrix = Trans_prob__matrix[sim][0]
        original_sum = Trans_prob__matrix[sim][1]
        # print(sim)
        simulated_values.append(testing_DTMC_M2(TP_matrix,original_sum, NHC_total_cases))

        
    # return TP_matrix_actuall_row_sum
    return simulated_values
    
    testmydtmc_M2_1K = simulate_DTMC_M2(Stoc_matrix_sum_1K, NHC_total_cases)
    testmydtmc_M2_10K = simulate_DTMC_M2(Stoc_matrix_sum_10K, NHC_total_cases)
    testmydtmc_M2_100K = simulate_DTMC_M2(Stoc_matrix_sum_100K, NHC_total_cases)
    
    # Use the Threshold function to find only the 10% of the lowest distances of the simulation runs

def threshold_acceptance(value):

    l = np.array(value[0])
    threshold = np.percentile(l, 10) # calculate the 10th percentile
    a = l[l < np.percentile(l, 10)] # Filter the list.

    # print(type(a))

    index_list = []
    for i in range(len(a)):
        index_list.append(np.where(l == a[i]))

    index_final_list = []
    for i in range(len(index_list)):
        index_final_list.append(index_list[i][0][0])

    # print(index_final_list)

    C_accepted = []
    for index in index_final_list:
        C_accepted.append(value[1][index])
    
    actual_sum_accepted = []
    for index in index_final_list:
        actual_sum_accepted.append(value[2][index])

    #     print (index)
    # print(C_accepted)
    return [a, np.array(C_accepted),actual_sum_accepted]


def POSTERIOR(observed_values, simulated_values):
    
    x = observed_values
    y = simulated_values
    length = len(y)

    C_obs = np.array(x[1][0])
    R_obs = np.array(x[2][0])
    D_obs = np.array(x[3][0])

    C_distances = [[],[],[]]
    R_distances = [[],[],[]]
    D_distances = [[],[],[]]
    
   
    for i in range(0, length):

        C_dist = np.linalg.norm(C_obs - np.array(y[i][0][0][1])) # calculate distance
        C_distances[0].append(C_dist)
        C_distances[1].append(y[i][0][0][4])
        C_distances[2].append(y[i][1][0][0])

        R_dist = np.linalg.norm(R_obs - np.array(y[i][0][0][2])) # calculate distance
        R_distances[0].append(R_dist)
        R_distances[1].append(y[i][0][0][4])
        R_distances[2].append(y[i][1][0][0])

        D_dist = np.linalg.norm(D_obs - np.array(y[i][0][0][3])) # calculate distance
        D_distances[0].append(D_dist)
        D_distances[1].append(y[i][0][0][4])
        D_distances[2].append(y[i][1][0][0])
    
    # Use the Threshold function 
    C_accepted = threshold_acceptance(C_distances) 
    R_accepted = threshold_acceptance(R_distances)
    D_accepted = threshold_acceptance(D_distances)
    
    return C_accepted, R_accepted, D_accepted

M2_value_1K = POSTERIOR(observed_values, testmydtmc_M2_1K)
M2_value_10K = POSTERIOR(observed_values, testmydtmc_M2_10K)
M2_value_100K = POSTERIOR(observed_values, testmydtmc_M2_100K)
# M2_value_1000K = POSTERIOR(observed_values, testmydtmc_M2)

def accepted_TMatrix(M2_value_xK):   

    '''
    args:
        M2_value_xK : List that contains accepted EU distance, TP MAtrix and Original Sum
    return:
        V :  consolidated list of TP_Matrix and Original Sum for Simulation Prediction
    '''

    v = [[],[]]
    v[0].extend(M2_value_xK[0][1])
    v[0].extend(M2_value_xK[1][1])
    v[0].extend(M2_value_xK[2][1])

    v[1].extend(M2_value_xK[0][2])
    v[1].extend(M2_value_xK[1][2])
    v[1].extend(M2_value_xK[2][2])

    return v
   
accepted_TMatrix_simulation_1K = accepted_TMatrix(M2_value_1K)
accepted_TMatrix_simulation_10K = accepted_TMatrix(M2_value_10K)
accepted_TMatrix_simulation_100K = accepted_TMatrix(M2_value_100K)

# np.matrix(accepted_TMatrix_simulation_1K[0][0])

def prediction_testing_DTMC_M2(Transition_matrix, NHC_List):
 
    '''
    args:
    NHC_List (List) = Generated by Log Function
    Total_simulations = No of simulations to be conducted

    returns:
    DTMC_predicted_values (NumPy array) = Predicted values, Transitions Probability Matrix

    '''
    DTMC_predicted_values = []

    DTMC_sub_array = [] # [Array[*Array*]]
    # DTMC_sub_array_2 = []

    for i in range(4):
        DTMC_sub_array.append([])     #creating list inside list

    value = NHC_List[0]
    current_state = np.matrix([value, 0, 0, 0])
    current_state = np.squeeze(np.asarray(current_state))   

    DTMC_sub_array[0].append(current_state[0])
    DTMC_sub_array[1].append(current_state[1])
    DTMC_sub_array[2].append(current_state[2])
    DTMC_sub_array[3].append(current_state[3])

    # iteration 1
    n_state = current_state * Transition_matrix
    n_state = np.squeeze(np.asarray(n_state)) 

    DTMC_sub_array[0].append(NHC_List[1])
    DTMC_sub_array[1].append(n_state[1])
    DTMC_sub_array[2].append(n_state[2])
    DTMC_sub_array[3].append(n_state[3])

    for i in range(1, len(NHC_List)):

        current_state = np.matrix([NHC_List[i], DTMC_sub_array[1][i-1], DTMC_sub_array[2][i-1], DTMC_sub_array[3][i-1]])
        new_state = current_state * Transition_matrix #Markov Chain
        new_state = np.squeeze(np.asarray(new_state)) 

        DTMC_sub_array[0].append(NHC_List[i])
        DTMC_sub_array[1].append(new_state[1])
        DTMC_sub_array[2].append(new_state[2])
        DTMC_sub_array[3].append(new_state[3])

    # Subarrays and respective Transition Probabilities merged
    Transition_matrix = np.squeeze(np.asarray(Transition_matrix))
    # DTMC_sub_array.append(Transition_matrix)
    # DTMC_sub_array_2.append(original_sum)

    DTMC_predicted_values.append(DTMC_sub_array)
    # DTMC_predicted_values[1].append(DTMC_sub_array_2)

    return DTMC_predicted_values


def prediction_simulate_DTMC_M2(Trans_prob__matrix, NHC_total_cases):
    '''
    args:
        NHC_total_cases (List)  = Total New Hospitalization Cases 
        Trans_Prob_Matrix (int) = Accepted Transition Matirx for prediction

    returns:
       simulated_values : predicted values for accepted Transition Matrix
    '''
    
    simulated_values = []

    # A1 = np.random.uniform(0.70,0.90,total_sims)
    # A2 = np.random.uniform(0.30,0.70,total_sims)

    # row1 = Trans_Matrix(A1)
    # row2 = Trans_Matrix(A2)

        # load additional module

    for sim in range(0, len(Trans_prob__matrix[0])):

        # print(sim)
        # print(len(Trans_prob__matrix))
        TP_matrix = np.matrix(Trans_prob__matrix[0][sim])
        # original_sum = Trans_prob__matrix[1][sim]
        # print(sim)
        simulated_values.append(prediction_testing_DTMC_M2(TP_matrix, NHC_total_cases))

        
    # return TP_matrix_actuall_row_sum
    return simulated_values

predicted_1K = prediction_simulate_DTMC_M2(accepted_TMatrix_simulation_1K, NHC_total_cases)
predicted_10K = prediction_simulate_DTMC_M2(accepted_TMatrix_simulation_10K, NHC_total_cases)
predicted_100K = prediction_simulate_DTMC_M2(accepted_TMatrix_simulation_10K, NHC_total_cases)

y_predicted_H, y_predicted_C = [], []
y_predicted_R, y_predicted_D = [], []

for i in range(len(predicted_1K)):

    y_predicted_H.append(predicted_1K[i][0][0])
    y_predicted_C.append(predicted_1K[i][0][1])
    y_predicted_R.append(predicted_1K[i][0][2])
    y_predicted_D.append(predicted_1K[i][0][3])
    
    
 from sklearn.metrics import mean_absolute_error


def MAE(y_observed, y_predicted):

    y_observed_H = y_observed[0][2:]
    y_observed_C = y_observed[1][2:] 
    y_observed_R = y_observed[2][2:] 
    y_observed_D = y_observed[3][2:]

    MAE_H, MAE_C, MAE_R, MAE_D = [],[],[],[]

    for i in range(len(y_predicted)):
        y_predicted_H = y_predicted[i][0][0]
        y_predicted_C = y_predicted[i][0][1]
        y_predicted_R = y_predicted[i][0][2]
        y_predicted_D = y_predicted[i][0][3]

       
        # MAE_H.append(mape(y_observed_H, y_predicted_H))
        # MAE_C.append(mape(y_observed_C, y_predicted_C))
        # MAE_R.append(mape(y_observed_R, y_predicted_R))
        # MAE_D.append(mape(y_observed_D, y_predicted_D))
            
        MAE_H.append(mean_absolute_error(y_observed_H, y_predicted_H))
        MAE_C.append(mean_absolute_error(y_observed_C, y_predicted_C))
        MAE_R.append(mean_absolute_error(y_observed_R, y_predicted_R))
        MAE_D.append(mean_absolute_error(y_observed_D, y_predicted_D))
        

    

    return[MAE_H, MAE_C, MAE_R, MAE_D]

MAE_1K = MAE(observed_values_1, predicted_1K)
MAE_10K = MAE(observed_values_1, predicted_10K)
MAE_100K = MAE(observed_values_1, predicted_100K)

least_MAE_C_100K = []
for i in range(len(MAE_100K[0])):
    if MAE_100K[1][i] < 300:
        least_MAE_C_100K.append(MAE_100K[1][i])

def least_MAE_calc(MAE_value_list,index_number,threshold):
    
    final_list = []
    for i in range(len(MAE_value_list[0])):
        if MAE_value_list[index_number][i] < threshold:
            final_list.append(MAE_value_list[index_number][i])
    return final_list
   
   # For 1 K

least_C_1K = least_MAE_calc(MAE_1K,1,1000)
least_R_1K = least_MAE_calc(MAE_1K,2,1000)
least_D_1K = least_MAE_calc(MAE_1K,3,1000)


# For 10 K

least_C_10K = least_MAE_calc(MAE_10K,1,10000)
least_R_10K = least_MAE_calc(MAE_10K,2,10000)
least_D_10K = least_MAE_calc(MAE_10K,3,10000)

# For 100 K

least_C_100K = least_MAE_calc(MAE_100K,1,500)
least_R_100K = least_MAE_calc(MAE_100K,2,15000)
least_D_100K = least_MAE_calc(MAE_100K,3,10000)

# least_D_100K


l = np.array(least_D_100K)
threshold = np.percentile(l, 1) # calculate the 10th percentile
a = l[l < np.percentile(l, 1)] # Filter the list.


# For Critical Care
'''
M2_C_accepted[0] : Accepted Euclidean Distances
M2_C_accepted[1] : Transition Probability Matrix
M2_C_accepted[2] : Actual Sum of the probabilities of the row
'''
# FOR 1K SIMULATIONS:
M2_C_accepted_1K = M2_value_1K[0]
M2_R_accepted_1K = M2_value_1K[1]
M2_D_accepted_1K = M2_value_1K[2]

# FOR 1K SIMULATIONS:
M2_C_accepted_10K = M2_value_10K[0]
M2_R_accepted_10K = M2_value_10K[1]
M2_D_accepted_10K = M2_value_10K[2]

# FOR 1K SIMULATIONS:
M2_C_accepted_100K = M2_value_100K[0]
M2_R_accepted_100K = M2_value_100K[1]
M2_D_accepted_100K = M2_value_100K[2]

# M2_C_accepted_1K

# To plot the Posterior Distribution

'''
Multiply the 1st and 2nd row of Transition Probability Matrix with the Actual sum of the probabilities
i.e.,   M2_C_accepeted[1] x M2_C_accepeted[2]
'''
def Posterior_Multiply(M2_C_accepted, M2_R_accepted, M2_D_accepted):
    
    '''
    args:
        M2_C/R/D_accepted[1] : Transition Probability Matrix
        M2_C/R/D_accepted[2] : Actual Sum of the probabilities of the row
    returns:
         multiplied_C/R/D_1_Post: First Row Values (PHH, PCH, PRH, PDH) 
         multiplied_C/R/D_2_Post: First Row Values (PHC, PCC, PRC, PDC)
    '''
    length = len(M2_C_accepted[0])

    multiplied_C1_Post, multiplied_R1_Post, multiplied_D1_Post = [],[],[]
    multiplied_C2_Post, multiplied_R2_Post, multiplied_D2_Post = [],[],[]

    for i in range(length):
        # print(i)
        multiplied_C1_Post.append(M2_C_accepted[1][i][0] * M2_C_accepted[2][i][0])
        multiplied_R1_Post.append(M2_R_accepted[1][i][0] * M2_R_accepted[2][i][0])
        multiplied_D1_Post.append(M2_D_accepted[1][i][0] * M2_D_accepted[2][i][0])

        multiplied_C2_Post.append(M2_C_accepted[1][i][1] * M2_C_accepted[2][i][1])
        multiplied_R2_Post.append(M2_R_accepted[1][i][1] * M2_R_accepted[2][i][1])
        multiplied_D2_Post.append(M2_D_accepted[1][i][1] * M2_D_accepted[2][i][1])

    Post_HH, Post_CH, Post_RH, Post_DH = [], [], [], []
    Post_HC, Post_CC, Post_RC, Post_DC = [], [], [], [] 

    
    for i in range(length):
        # print(i)
        Post_HH.append(multiplied_C1_Post[i][0])
        Post_HH.append(multiplied_R1_Post[i][0])
        Post_HH.append(multiplied_D1_Post[i][0])

        Post_CH.append(multiplied_C1_Post[i][1])
        Post_CH.append(multiplied_R1_Post[i][1])
        Post_CH.append(multiplied_D1_Post[i][1])

        Post_RH.append(multiplied_C1_Post[i][2])
        Post_RH.append(multiplied_R1_Post[i][2])
        Post_RH.append(multiplied_D1_Post[i][2])

        Post_DH.append(multiplied_C1_Post[i][3])
        Post_DH.append(multiplied_R1_Post[i][3])
        Post_DH.append(multiplied_D1_Post[i][3])

        Post_HC.append(multiplied_C2_Post[i][0])
        Post_HC.append(multiplied_R2_Post[i][0])
        Post_HC.append(multiplied_D2_Post[i][0])

        Post_CC.append(multiplied_C2_Post[i][1])
        Post_CC.append(multiplied_R2_Post[i][1])
        Post_CC.append(multiplied_D2_Post[i][1])

        Post_DC.append(multiplied_C2_Post[i][3])
        Post_DC.append(multiplied_R2_Post[i][3])
        Post_DC.append(multiplied_D2_Post[i][3])

        Post_RC.append(multiplied_C2_Post[i][2])
        Post_RC.append(multiplied_R2_Post[i][2])
        Post_RC.append(multiplied_D2_Post[i][2])

    return Post_HH, Post_CH, Post_RH, Post_DH, Post_HC, Post_CC, Post_RC, Post_DC
    




    # Post_HH, Post_CH, Post_RH, Post_DH = [], [], [], []
    # Post_HC, Post_CC, Post_RC, Post_DC = [], [], [], [] 
    # for i in range(len(length)):
        
    #     #row1
    #     Post_HH.append(multiplied_C_Post[i][0])
    #     Post_CH.append(multiplied_C_Post[i][1]) 
    #     Post_RH.append(multiplied_C_Post[i][2]) 
    #     Post_DH.append(multiplied_C_Post[i][3]) 

    #     # row2
    #     Post_HC.append(multiplied_C_Post[i][0])
    #     Post_CC.append(multiplied_C_Post[i][1]) 
    #     Post_RC.append(multiplied_C_Post[i][2]) 
    #     Post_DC.append(multiplied_C_Post[i][3])

    # posteior_all_parameters = [ Post_HH, Post_CH, Post_RH, Post_DH,
    #                             Post_HC, Post_CC, Post_RC, Post_DC ] 

# P_HH = []
# for i in range(100):
#     prh.append(multiplied_C_Post[i][2])
    
 
 m = Posterior_Multiply_2(M2_C_accepted_1K, M2_R_accepted_1K, M2_D_accepted_1K)
post_multy_1K = Posterior_Multiply(M2_C_accepted_1K, M2_R_accepted_1K, M2_D_accepted_1K)
post_multy_10K = Posterior_Multiply(M2_C_accepted_10K, M2_R_accepted_10K, M2_D_accepted_10K)
post_multy_100K = Posterior_Multiply(M2_C_accepted_100K, M2_R_accepted_100K, M2_D_accepted_100K)

def plot_posterior(probability_list,title):

    plt.hist(probability_list, bins = 35, alpha = 0.5)    
    plt.title(title)
    plt.xlabel('Probabilities')
    plt.ylabel('Count')
    plt.legend()
    plt.rcParams['figure.figsize']= [8,5]
    plt.savefig('/Users/pritishsadiga/Desktop/plot.png')
    return plt.show()
    
def PlOT_POSTERIOR_SIM(posterior):     

    '''
    args:
        posterior (array) : Array of all the posterior parameters
    returns:
        posterior distribution (graph) : posterior distribution of all accepted parameters
    '''

    # Plotting Posterior Distribution of all accepted parameters of Transition Matrix ROW 1
    HH_post = plot_posterior(posterior[0],'Posterior: P(H|H)')
    CH_post = plot_posterior(posterior[1],'Posterior: P(C|H)')
    RH_post = plot_posterior(posterior[2],'Posterior: P(R|H)')
    DH_post = plot_posterior(posterior[3],'Posterior: P(D|H)')

    # Plotting Posterior Distribution of all accepted parameters of Transition Matrix ROW 2
    HC_post = plot_posterior(posterior[4],'Posterior: P(H|C)')
    CC_post = plot_posterior(posterior[5],'Posterior: P(C|C)')
    RC_post = plot_posterior(posterior[6],'Posterior: P(R|C)')
    DC_post = plot_posterior(posterior[7],'Posterior: P(D|C)')

    return HH_post, CH_post, RH_post, DH_post, HC_post, CC_post, RC_post, DC_post
   
   
PlOT_POSTERIOR_SIM(post_multy_1K)
PlOT_POSTERIOR_SIM(post_multy_10K)
PlOT_POSTERIOR_SIM(post_multy_100K)
