%load_ext autotime
import numpy as np
import pandas as pd
import math
from scipy import random
from scipy import stats
import statistics
import matplotlib.pyplot as plt
import pickle

# Data Type 1 (with cumulative values)
data_1 = pd.read_csv('//Users/pritishsadiga/Desktop/MS_Thesis/Data_Testing/data_test.csv')
# Data Type 2
data = pd.read_csv('/Users/pritishsadiga/Desktop/MS_Thesis/Data_Testing/data.csv')

Observed_values_1 = [data_1['hospitalizedCurrently'].tolist(),
                    data_1['CriticalCurrently'].tolist(),
                    data_1['RecoveredCurrently'].tolist(),
                    data_1['DeathCurrently'].tolist()]

Observed_values_1 = [data_1['hospitalizedCurrently'].tolist(),
                    data_1['CriticalCurrently'].tolist(),
                    data_1['RecoveredCurrently'].tolist(),
                    data_1['DeathCurrently'].tolist()]

# ---------------------------------------------------------------To Fill the first N - Zero Elements with values from a an Exponential function ---------------------------------------------------------------

def fill_missing_values(list_values, rate):

    count_index = 0
    for i in range(len(list_values)):
    
        if list_values[i] == 0:
            count_index += 1

    index_value = count_index
    days = index_value
    crit_cases = []
    crit_rate = rate
    current_value = 1
    for i in range(days):
        next_value = current_value * math.exp(rate)
        crit_cases.append(next_value)
        current_value = next_value

    list_1 = [round(num) for num in crit_cases]
    # print(len(list_1))
    list_2 = list_values[index_value + 1:]
    # print(len(list_2))
    Final_Critical_cases = list_1 + list_2
    
    return Final_Critical_cases



# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

def NHC_calculator(Missing_Hospitalization_Cases):    
    
    rate = 0
    rate_count = 0
    NHC_avg_rate = [] 

    for i in range(0, len(Missing_Hospitalization_Cases) - 1):

        rate += math.log(Missing_Hospitalization_Cases[i + 1] / Missing_Hospitalization_Cases[i])
        rate_count += 1        

        if rate_count == 7:     #once count is equal to a week, caluclate average rate
            NHC_avg_rate.append(rate / 7)
            rate = 0
            rate_count = 0
    # print(NHC_avg_rate)

    # To generate NHC based on 1-week rate change
    day = 0
    NHC_cases = []
    previous_day = 1

    for i in range(0, len(Missing_Hospitalization_Cases) - 1):
  
        avg_rate = NHC_avg_rate[day // 7]
        new_day = previous_day * math.exp(avg_rate)
        NHC_cases.append(new_day)
        previous_day = new_day

        day += 1
        if day // 7 >= len(NHC_avg_rate):
            break

    NHC_Cases = [round(num) for num in NHC_cases]

    return NHC_Cases

    # -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

def Trans_Matrix_II(total_values):
  
    P_HH = (np.random.uniform(0.50, 0.70, total_values)).round(4)
    P_CH = (np.random.uniform(0.30, 0.50, total_values)).round(4)
    P_RH = (np.random.uniform(0.0, 0.30, total_values)).round(4)
    P_DH = (np.random.uniform(0.0, 0.10, total_values)).round(4)

    P_HC = (np.random.uniform(0.50, 0.70, total_values)).round(4)
    P_CC = (np.random.uniform(0.30, 0.50, total_values)).round(4)
    P_RC = (np.random.uniform(0.0, 0.30, total_values)).round(4)
    P_DC = (np.random.uniform(0.0, 0.10, total_values)).round(4)

    return [   P_HH, P_CH, P_RH, P_DH, 
               P_HC, P_CC, P_RC, P_DC    ]

# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------


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
# prior_plots_1000K = PlOT_PRIOR_SIM(prior_prob_1000K)

# To coi

# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------


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

# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------


def DTMC_model(Transition_matrix, original_sum, NHC_List):
 
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


# Simulation of MARKOV CHAIN

def DTMC_simulation(Trans_prob__matrix, NHC_total_cases):
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

    for sim in range(0, len(Trans_prob__matrix)):

        # print(sim)
        # print(len(Trans_prob__matrix))
        TP_matrix = Trans_prob__matrix[sim][0]
        original_sum = Trans_prob__matrix[sim][1]
        # print(sim)
        simulated_values.append( DTMC_model (TP_matrix,original_sum, NHC_total_cases))

        
    # return TP_matrix_actuall_row_sum
    return simulated_values
    
# -------------------------------------------------------------- Saving the File --------------------------------------------------------------

def file_txt_store(file_list, sim_name, date):   
    with open('/Users/pritishsadiga/Desktop/MS_Thesis/prior'+ str(sim_name) + str(date) + '.txt' , 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(file_list, filehandle)
        return print('File_stored')
    # 
# -------------------------------------------------------------- Use the Threshold function to find only the 10% of the lowest distances of the simulation run--------------------------------------------------------------


def threshold_acceptance(value):

    l = np.array(value[0])
    threshold = np.percentile(l, 10) # calculate the 10th percentile
    a = l[l < np.percentile(l, 10)] # Filter the list.

    # a = []
    # for i in range(len(value)):
        
    #     if value[0][i] < 10000:
    #         a.append(value[i])

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

# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

def accepted_TMatrix(M2_value_xK):   

    '''
    args:
        DTMC_Posterior_xK : List that contains accepted EU distance, TP MAtrix and Original Sum
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

# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

def Posterior_simulation(Trans_prob__matrix, NHC_total_case):
    '''
    args:
       trans_prob_matrix : Transition Matrix from Posterior
       NHC_total_cases: New Hospitalization cases
    returns:
        simulated_values : List containing DTMC predicted values, transition matrix and original sum 
    '''

    posterior_simulation_values = []
    for sim in range(0, len(Trans_prob__matrix[0])):

        TP_matrix = np.matrix(Trans_prob__matrix[0][sim])
        original_sum = Trans_prob__matrix[1][sim]
        
        posterior_simulation_values.append(DTMC_model(TP_matrix,original_sum, NHC_total_case))

    # return TP_matrix_actuall_row_sum
    return posterior_simulation_values




# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

from sklearn.metrics import mean_absolute_error
def Mean_Absolute_Error(y_observed, y_predicted):

    y_observed_H = y_observed[0][4:]
    y_observed_C = y_observed[1][4:] 
    y_observed_R = y_observed[2][4:] 
    y_observed_D = y_observed[3][4:]

    MAE_H, MAE_C, MAE_R, MAE_D = [],[],[],[]
    MAE_H_1, MAE_C_1, MAE_R_1, MAE_D_1 = [], [], [], []

    for i in range(len(y_predicted)):
        y_predicted_H = y_predicted[i][0][0][0]
        y_predicted_C = y_predicted[i][0][0][1]
        y_predicted_R = y_predicted[i][0][0][2]
        y_predicted_D = y_predicted[i][0][0][3]

        MAE_H.append(mean_absolute_error(y_observed_H, y_predicted_H))
        MAE_C.append(mean_absolute_error(y_observed_C, y_predicted_C))
        MAE_R.append(mean_absolute_error(y_observed_R, y_predicted_R))
        MAE_D.append(mean_absolute_error(y_observed_D, y_predicted_D))

        MAE_H_1.append(mean_absolute_error(y_observed_H, y_predicted_H)/(sum(y_observed_H)/len(y_observed_H)))
        MAE_C_1.append(mean_absolute_error(y_observed_C, y_predicted_C)/(sum(y_observed_C)/len(y_observed_C)))
        MAE_R_1.append(mean_absolute_error(y_observed_R, y_predicted_R)/(sum(y_observed_R)/len(y_observed_R)))
        MAE_D_1.append(mean_absolute_error(y_observed_D, y_predicted_D)/(sum(y_observed_D)/len(y_observed_D)))
        
    return[[MAE_H, MAE_C, MAE_R, MAE_D],[MAE_H_1, MAE_C_1, MAE_R_1, MAE_D_1]]


def least_MAE_calc(MAE_value_list,index_number,threshold):
    
    final_list = []
    for i in range(len(MAE_value_list[0])):
        if MAE_value_list[index_number][i] < threshold:
            final_list.append(MAE_value_list[index_number][i])
    return final_list


from sklearn.metrics import mean_absolute_percentage_error
def Mean_Absolute_Percentage_Error(y_observed, y_predicted):

    y_observed_H = y_observed[0][4:]
    y_observed_C = y_observed[1][4:] 
    y_observed_R = y_observed[2][4:] 
    y_observed_D = y_observed[3][4:]

    MAPE_H, MAPE_C, MAPE_R, MAPE_D = [],[],[],[]

    for i in range(len(y_predicted)):
        y_predicted_H = y_predicted[i][0][0][0]
        y_predicted_C = y_predicted[i][0][0][1]
        y_predicted_R = y_predicted[i][0][0][2]
        y_predicted_D = y_predicted[i][0][0][3]

        MAPE_H.append(mean_absolute_percentage_error(y_observed_H, y_predicted_H))
        MAPE_C.append(mean_absolute_percentage_error(y_observed_C, y_predicted_C))
        MAPE_R.append(mean_absolute_percentage_error(y_observed_R, y_predicted_R))
        MAPE_D.append(mean_absolute_percentage_error(y_observed_D, y_predicted_D))

    return [[MAPE_H, MAPE_C, MAPE_R, MAPE_D]]

from sklearn.metrics import mean_squared_error
import math
def Root_Mean_Squared_Error(y_observed, y_predicted):

    y_observed_H = y_observed[0][4:]
    y_observed_C = y_observed[1][4:] 
    y_observed_R = y_observed[2][4:] 
    y_observed_D = y_observed[3][4:]

    RMSE_H, RMSE_C, RMSE_R, RMSE_D = [],[],[],[]

    for i in range(len(y_predicted)):
        y_predicted_H = y_predicted[i][0][0][0]
        y_predicted_C = y_predicted[i][0][0][1]
        y_predicted_R = y_predicted[i][0][0][2]
        y_predicted_D = y_predicted[i][0][0][3]

        MSE_H = mean_absolute_error(y_observed_H, y_predicted_H)
        RMSE_0 = math.sqrt(MSE_H)
        RMSE_H.append(RMSE_0)

        MSE_C = mean_absolute_error(y_observed_C, y_predicted_C)
        RMSE_1 = math.sqrt(MSE_C)
        RMSE_C.append(RMSE_1)

        MSE_R = mean_absolute_error(y_observed_R, y_predicted_R)
        RMSE_2 = math.sqrt(MSE_R)
        RMSE_R.append(RMSE_2)

        MSE_D = mean_absolute_error(y_observed_D, y_predicted_D)
        RMSE_3 = math.sqrt(MSE_D)
        RMSE_D.append(RMSE_3)

        
        
    return [RMSE_H, RMSE_C, RMSE_R, RMSE_D]

# Fucntion to choose the  best MAE values
def lowest_error_values(state_values, limit1, limit2, limit3):

    '''
    args: 
        state_values: A list containing the posterior values of different states

    returns:
        accepted_MAE / MAPE = Best accepted values
    '''
    
    H, C, R, D = [[],[]], [[],[]], [[],[]], [[],[]]
    for i in range(0, len(state_values[0])):
        
        # print(state_values[2]) 
        if state_values[0][i] <= 0.2:
            H[0].append(state_values[0][i])
            H[1].append((state_values[0]).index(state_values[0][i]))
            
        if state_values[1][i] <= limit1:
            C[0].append(state_values[1][i])
            C[1].append((state_values[1]).index(state_values[1][i]))

        if state_values[2][i] <= limit2:
            R[0].append(state_values[2][i])
            R[1].append((state_values[2]).index(state_values[2][i]))

        if state_values[3][i] <= limit3:
            D[0].append(state_values[3][i])
            D[1].append((state_values[3]).index(state_values[3][i]))
        
    lowest_error_values = [H, C, R, D]

    return lowest_error_values

# Filtering best MAE/Mean (below 0.1 - 0.2) and MAPE values (below 0.20 = Good, below 0.10 = Excellent )
def Filtered_errors(MAE, MAPE, RMSE):
    '''
    args: 
        MAE, MAPE, RMSE : List of all the Errors from 1K, 10K and 100K simulations
    returns:
        List: A list containing all filtered error values for the 1K, 10K and 100K simulations
        List[0][0] = MAE
        List[0][1][0] = MAE / Mean
        List[0][1][1] = Indices
        List[1] = MAPE
        List[2] = RMSE
    '''
    Filtered_MAE = lowest_error_values(MAE,0.25,0.25,0.25)
    Filtered_MAPE = lowest_error_values(MAPE,0.25,0.25,0.25)
    Filtered_RMSE = lowest_error_values(RMSE,100,100,100)

    return [Filtered_MAE, Filtered_MAPE, Filtered_RMSE]
    
# ----------------- 12.6 Finding the Index of best RMSE, MAE, MAE/Mean and MAPE values and creating an Interesection (A n B n C) List  ----------------------------

# Matching the indexes of the values of best MAE / Mean and MAPE values
def error_mapping_intersection(MAE, MAPE, RMSE, Filtered_Error_MAE, Filtered_Error_MAPE, Filtered_Error_RMSE, Posterior_predicted):
    '''
    args:
        MAE, MAPE, RMSE : Error Values List
        Filtered_Error_MAE, MAPE, RMSE : Filterest List obtained from 'FUNCTION - Filtered_errors'
        Posterior_predicted: Predicted Values o
    return:
        List [0] : Predicted Values (with Filtered Errors) for Graphs
            List[0][0] : Predicted Values (based on Errors over Critical Care)
            List[0][1] : Predicted Values (based on Errors over Recovered Cases)
            List[0][2] : Predicted Values (based on Errors over Death Cases)

        List [1] : Indicies
            List [1][0] : Common Indicies of Values of Posterior Prediction with least errors
            List [1][1] : Unique Indices of Values of Posterior Prediction with least errors

        List [2] : Results
            List [2][0] : Transiton Probabilitiy Matrix
            List [2][1] : MAE Values
            List [2][2] : Scaled MAE Values
            List [2][3] : RMSE Values
            List [2][4] : MAPE % Values
    '''

    Common_indices_C = list(set(Filtered_Error_MAE[0][1][1]) & set(Filtered_Error_MAPE[1][1][1]) & set(Filtered_Error_RMSE[2][1][1]))
    Common_indices_R = list(set(Filtered_Error_MAE[0][2][1]) & set(Filtered_Error_MAPE[1][2][1]) & set(Filtered_Error_RMSE[2][2][1]))
    Common_indices_D = list(set(Filtered_Error_MAE[0][3][1]) & set(Filtered_Error_MAPE[1][3][1]) & set(Filtered_Error_RMSE[2][3][1]))

    All_Common_indices = Common_indices_C + Common_indices_R + Common_indices_D
    All_Common_indices_set = list(set(All_Common_indices))


    # Mapping the index values towards the Predicted Values of Posterior Distribution to identify the acceptable Transition Matrix of the Model
    Predicted_C,Predicted_R,Predicted_D = [],[],[]

    for i in range(len(Common_indices_C)):
        Predicted_C.append(Posterior_predicted_100K[Common_indices_C[i]])

    for i in range(len(Common_indices_R)):
        Predicted_R.append(Posterior_predicted_100K[Common_indices_R[i]])

    for i in range(len(Common_indices_D)):
        Predicted_D.append(Posterior_predicted_100K[Common_indices_D[i]])

    
    # Extracting the Transition Probabiltiies from the common indices from the 
    TMatrix_common = []
    TMI = All_Common_indices
    for i in range(0, len(TMI)):
        TMatrix_common.append(np.matrix(Posterior_predicted[TMI[i]][0][0][4]).round(4))


    MAE_final_list_from_C, MAE_final_list_from_R, MAE_final_list_from_D = [],[],[] # MAE
    Scaled_MAE_final_list_from_C, Scaled_MAE_final_list_from_R, Scaled_MAE_final_list_from_D = [],[],[] # MAE/Meam
    MAPE_final_list_from_C, MAPE_final_list_from_R, MAPE_final_list_from_D = [],[],[] # MAPE
    RMSE_final_list_from_C, RMSE_final_list_from_R, RMSE_final_list_from_D = [],[],[] # RMSE

    for i in range(len(Common_indices_C)):
        MAE_final_list_from_C.append(MAE[0][1][Common_indices_C[i]])
        Scaled_MAE_final_list_from_C.append(MAE[1][1][Common_indices_C[i]])
        MAPE_final_list_from_C.append(MAPE[0][1][Common_indices_C[i]])
        RMSE_final_list_from_C.append(RMSE[1][Common_indices_C[i]])
       

    for i in range(len(Common_indices_R)):
        MAE_final_list_from_R.append(MAE[0][2][Common_indices_R[i]])
        Scaled_MAE_final_list_from_R.append(MAE[1][2][Common_indices_R[i]])
        MAPE_final_list_from_R.append(MAPE[0][2][Common_indices_R[i]])
        RMSE_final_list_from_R.append(RMSE_100K[2][Common_indices_R[i]])

    for i in range(len(Common_indices_D)):
        MAE_final_list_from_D.append(MAE[0][3][Common_indices_D[i]])
        Scaled_MAE_final_list_from_D.append(MAE[1][3][Common_indices_D[i]])
        MAPE_final_list_from_D.append(MAPE[0][3][Common_indices_D[i]])
        RMSE_final_list_from_D.append(RMSE_100K[3][Common_indices_D[i]])


    # Sum of all Mapped Values for each state

    MAE_all_mapped_mae_mape_values = MAE_final_list_from_C + MAE_final_list_from_R + MAE_final_list_from_D
    Scaled_MAE_all_mapped_mae_mape_values = Scaled_MAE_final_list_from_C + Scaled_MAE_final_list_from_R + Scaled_MAE_final_list_from_D
    RMSE_all_mapped_mae_mape_values = RMSE_final_list_from_C + RMSE_final_list_from_R + RMSE_final_list_from_D

    MAPE_all_mapped_mae_mape_values = MAPE_final_list_from_C + MAPE_final_list_from_R + MAPE_final_list_from_D
    MAPE_all_mapped_mae_mape_values_100 = []
    for i in range(len(MAPE_all_mapped_mae_mape_values)):
        MAPE_all_mapped_mae_mape_values_100.append(MAPE_all_mapped_mae_mape_values[i] * 100)

    
    return [[Predicted_C, Predicted_R, Predicted_D], 
            [All_Common_indices, All_Common_indices_set], 
            [TMatrix_common, RMSE_all_mapped_mae_mape_values, MAE_all_mapped_mae_mape_values, Scaled_MAE_all_mapped_mae_mape_values, MAPE_all_mapped_mae_mape_values_100]]

# ---------------------------To Save the Results in CSV -----------------------------------------------

def Results_DF(results, FileName, condition):

    value_df = pd.DataFrame({'Transition Probability Matrix':  results[2][0] ,
                                'RMSE': results[2][1], 
                                'MAE': results[2][2], 
                                'MAE/Mean':results[2][3] , 
                                'MAPE %':results[2][4]})

    # condition = input("Enter your value (Save File / View DataFrame): ")
        
    if condition == 'Save File':
        value_df.to_csv('/Users/pritishsadiga/Desktop/MS_Thesis/' + f'{FileName}' + '.csv')
        return print('Your Results have been saved in a CSV')

    elif condition == 'View DataFrame':
            return value_df
    else:
        return print('Enter the correct value')



# -------------------------------------------------------------- To calculate the rate of increase of cases from March 1 to March 17--------------------------------------------------------------

# To plot the Posterior Distribution

def Posterior_Multiply(DMTC_Posterior_Accepted_1K):
    
 
    length = len(DMTC_Posterior_Accepted_1K[0])
    First_row, Second_Row = [],[]
    for i in range(length):
        
        First_row.append(DMTC_Posterior_Accepted_1K[0][i][0] * DMTC_Posterior_Accepted_1K[1][i][0])
        Second_Row.append(DMTC_Posterior_Accepted_1K[0][i][1] * DMTC_Posterior_Accepted_1K[1][i][1])
      
    Post_HH, Post_CH, Post_RH, Post_DH = [], [], [], []
    Post_HC, Post_CC, Post_RC, Post_DC = [], [], [], [] 

    for i in range(length):
        # print(i)
        Post_HH.append(First_row[i][0])
        Post_CH.append(First_row[i][1])
        Post_RH.append(First_row[i][2])
        Post_DH.append(First_row[i][3])
  
        Post_HC.append(Second_Row[i][0])
        Post_CC.append(Second_Row[i][1])
        Post_DC.append(Second_Row[i][3])
        Post_RC.append(Second_Row[i][2])

    return Post_HH, Post_CH, Post_RH, Post_DH, Post_HC, Post_CC, Post_RC, Post_DC
    
    
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

def plot_graph(col1, col2, col3, col4,col11, col12, col13, col14, title1, title2):

    
        plt.plot(col1,color='green',ls = '--',label= 'Predicted H')
        plt.plot(col2,color='red',ls = '--',label='Predicted C')
        plt.plot(col3,color='blue',ls = '--',label= 'Predicted R')
        plt.plot(col4,color='black',ls = '--',label= 'Predicted D')

        plt.plot(col11,color='green', label= 'Observed H')
        plt.plot(col12,color='red',label='Observed C')
        plt.plot(col13,color='blue',label= 'Observed R')
        plt.plot(col14,color='black',label= 'Observed D')

        plt.xlabel('Days')
        plt.ylabel('Cases')
        plt.legend()
        
        
        # tpmatrix = iteration value for graph number
        plt.title(f'{title1}' + ' ' + f'Graph' + ' ' + f'{tpmatrix}' + ':' + f'{title2}')
        
        plt.rcParams['figure.figsize']= [20,6]

        return plt.show()

def plot_hist(data_column, hist_title):
    x = data_column
    plt.title(hist_title)

    plt.hist(x, bins = 250)
    return plt.show()
# -----------------------------------------------------------------------------------------------------------------
   
### 1. To Fill All Missing data points in the Data Set 
   
# Call the function to get updated values from the original data
Missing_Hospitalization_Cases = fill_missing_values(Observed_values_1[0], 0.3855)
Missing_Critical_Cases = fill_missing_values(Observed_values_1[1], 0.29845)
Missing_Recovered_Cases = fill_missing_values(Observed_values_1[2], 0.30493)
Missing_Death_Cases = fill_missing_values(Observed_values_1[3], 0.09)


# List with missing values filled by exponential function
observed_values = [Missing_Hospitalization_Cases, Missing_Critical_Cases, Missing_Recovered_Cases, Missing_Death_Cases]   

def rate_calc(list_value):
    rate = 0
    rate_count = 0
    avg_rate = [] 

    for i in range(0, len(list_value) - 1):

        rate += math.log(list_value[i + 1] / list_value[i])
        rate_count += 1        

        if rate_count == 7:     #once count is equal to a week, caluclate average rate
            avg_rate.append(rate / 7)
            rate = 0
            rate_count = 0

    return avg_rate
   
 ### 2. New Hospitalization Cases
 
 NHC_total_case = NHC_calculator(Missing_Hospitalization_Cases)
 
 
 ### 3. Calculating Transition Matrix
 
 Prior_1K = Trans_Matrix_II(1000)
Prior_10K = Trans_Matrix_II(10000)
Prior_100K = Trans_Matrix_II(100000)

### 4. Prior Distributions

PlOT_PRIOR_SIM(Prior_1K)
PlOT_PRIOR_SIM(Prior_10K)
PlOT_PRIOR_SIM(Prior_100K)

### 5. Converting Transition Matrix to Stochastic Matrix

Stoc_matrix_sum_1K = STOCH_Matrix_original_sum(Prior_1K, 1000)
Stoc_matrix_sum_10K = STOCH_Matrix_original_sum(Prior_10K, 10000)
Stoc_matrix_sum_100K = STOCH_Matrix_original_sum(Prior_100K, 100000)

### 6. Saving the Values (If needed)

# Function to write files into TXT and To store the prior parameters of the transition probability matrix 

file_txt_store(Stoc_matrix_sum_1K, 'TMatrix1K', '03_11_21') 
file_txt_store(Stoc_matrix_sum_10K, 'TMatrix10K', '03_11_21')
file_txt_store(Stoc_matrix_sum_100K, 'TMatrix100K', '03_11_21')

### 7. Discrete Time Markov chain Model and Simulation upto 1K, 10K and 100K

DTMC_1K = DTMC_simulation(Stoc_matrix_sum_1K, NHC_total_case)
DTMC_10K = DTMC_simulation(Stoc_matrix_sum_10K, NHC_total_case)
DTMC_100K = DTMC_simulation(Stoc_matrix_sum_100K, NHC_total_case)

### 8. Approximate Bayesian Computation Rejection Algorithm

DTMC_Posterior_1K = POSTERIOR(observed_values, DTMC_1K)
DTMC_Posterior_10K = POSTERIOR(observed_values, DTMC_10K)
DTMC_Posterior_100K = POSTERIOR(observed_values, DTMC_100K)

### 9. Consolidating Accepted Values and their respective Transition Probabilities

DMTC_Posterior_Accepted_1K = accepted_TMatrix(DTMC_Posterior_1K)
DMTC_Posterior_Accepted_10K = accepted_TMatrix(DTMC_Posterior_10K)
DMTC_Posterior_Accepted_100K = accepted_TMatrix(DTMC_Posterior_100K)

### 10. Posterior Distribution

Posteiror_1K = Posterior_Multiply(DMTC_Posterior_Accepted_1K)
Posteiror_10K = Posterior_Multiply(DMTC_Posterior_Accepted_10K)
Posteiror_100K = Posterior_Multiply(DMTC_Posterior_Accepted_100K)

#Plotting the graphs
PlOT_POSTERIOR_SIM(Posteiror_1K)
PlOT_POSTERIOR_SIM(Posteiror_10K)
PlOT_POSTERIOR_SIM(Posteiror_100K)

### 11. Generate data using Posterior distribution

Posterior_predicted_1K = Posterior_simulation(DMTC_Posterior_Accepted_1K, NHC_total_case)
Posterior_predicted_10K = Posterior_simulation(DMTC_Posterior_Accepted_1K, NHC_total_case)
Posterior_predicted_100K = Posterior_simulation(DMTC_Posterior_Accepted_1K, NHC_total_case)

### 12. Calculating Mean Absolute Error (MAE)

MAE_1K = Mean_Absolute_Error(observed_values, Posterior_predicted_1K)
MAE_10K = Mean_Absolute_Error(observed_values, Posterior_predicted_10K)
MAE_100K = Mean_Absolute_Error(observed_values, Posterior_predicted_100K)

def plot_graph(col1, col2, col3, col4, graph_title):

    plt.plot(col1,color='green',label= 'H')
    plt.plot(col2,color='red',label='C')
    plt.plot(col3,color='blue',label= 'R')
    plt.plot(col4,color='black',label= 'D')
    plt.title(graph_title)
    plt.xlabel('Days')
    plt.ylabel('Cases')
    plt.legend()
    plt.rcParams['figure.figsize']= [20,6]

    return plt.show()

def plot_hist(data_column, hist_title):
    x = data_column
    plt.title(hist_title)
    plt.hist(x, bins = 250)
    return plt.show()
   
 
