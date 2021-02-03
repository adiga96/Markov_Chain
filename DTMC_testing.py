# Testing 1: Markov Chain Transition probabilities generator

row_1 = Trans_Matrix(0.80,0.90)
row_2 = Trans_Matrix(0.70,0.80)
row_3 = [0, 0,  1,  0]
row_4 = [0, 0,  0,  1]

Testing_TRANS_PROB  = np.matrix([row_1,row_2,row_3,row_4])
# Testing_TRANS_PROB 

# Testing 2:
'''
x = exp_simulation(min_rate ,max_rate , 1, 325, 40, y)
x = [round(num) for num in x]

observed_values = data_copy['hospitalized'][:INC_limit].fillna(0).dropna().tolist()
a = data_copy[data_copy['hospitalized'] != 0]
observed_values = a['hospitalized'][:40].tolist()
observed_values = [round(num) for num in observed_values]
'''
observed_values = data_copy['hospitalized'][:INC_limit].fillna(0).dropna().tolist()
a = data_copy[data_copy['hospitalized'] != 0]
observed_values = a['hospitalized'][:27].tolist()
# observed_values

# Testing 3:

max_rate = 0.1500
min_rate = 0.1480
 
inc_all_rates = increasing_rate_calculator(data)

testing_simulation = exp_simulation(max_rate, min_rate, 1000, 325, 27)
testing_KStest = KS_TEST(testing_simulation,observed_values)

total_accepted_values = len(testing_KStest)
total_accepted_values

# Testing 4: Using the DTMC Log Function predictor

b = testing_simulation[2][0]
DTMC_testing_prediction = testing_DTMC_log_increasing(Testing_TRANS_PROB, b)

plt.plot(DTMC_testing_prediction[0][0],color='green',label='Hospitalized')
plt.plot(DTMC_testing_prediction[0][1],color='red',label='Critical')
plt.plot(DTMC_testing_prediction[0][2],color='blue',label='Recovered')
plt.plot(DTMC_testing_prediction[0][3],color='black',label='Death')


plt.legend()
plt.rcParams['figure.figsize']= [20,6]

# Testing 5: 

# dec_min_rate = -0.03000
# dec_max_rate = -0.02000
# dec_all_rates = decreasing_rate_calculator(data)

testing_simulation_dec = exp_simulation(dec_min_rate,dec_max_rate,10000,18825,153)
testing_KStest_dec = KS_TEST(testing_simulation_dec,observed_values)
total_accepted_values_dec = len(testing_KStest_dec)
# total_accepted_values_dec

# Testing 5:

a = [round(num, 5) for num in testing_simulation_dec[2][0]]
testing_NHS_dec_ist = a
DTMC_testing_prediction_dec = testing_DTMC_log_increasing(Testing_TRANS_PROB, testing_NHS_dec_ist)
plt.plot(DTMC_testing_prediction_dec[0][0],color='green',label='Hospitalized')
plt.plot(DTMC_testing_prediction_dec[0][1],color='red',label='Critical')
plt.plot(DTMC_testing_prediction_dec[0][2],color='blue',label='Recovered')
plt.plot(DTMC_testing_prediction_dec[0][3],color='black',label='Death')


plt.legend()
plt.rcParams['figure.figsize']= [20,6]

