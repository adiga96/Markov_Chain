import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')
data_copy = data # create a copy of the data
hospitalized = data_copy['hospitalized'].tolist()


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

# To plot

x = np.linspace(0,223,224)
plt.scatter(x,values, label = 'NHC Rate data')
plt.plot(hospitalized, color = 'red', label= 'Observed Data')
plt.axvline(x=20, color ='green', label = 'lockdown')
plt.legend()
# plt.plot(hospitalized)
