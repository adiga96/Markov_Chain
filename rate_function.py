import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


data = pd.read_csv('/Users/pritishsadiga/Desktop/test.csv')
data_copy = data # create a copy of the data
hospitalized = data_copy['hospitalized'].tolist()


# To calculate the 7 days, 1 week rate of addition of NHC (New Hopsitalization Cases)
rate = 0
rate_count = 0
NHC_avg_rate = [] 

for i in range(0, len(hospitalized) - 1):

    rate += math.log(hospitalized[i + 1] / hospitalized[i])
    rate_count += 1        #count every iteration

    if rate_count == 7:     #once count is equal to a week, caluclate average rate
        NHC_avg_rate.append(rate / 7)
        rate = 0
        rate_count = 0
        
NHC_avg_rate = [round(num) for num in NHC_avg_rate]
# NHC_avg_rate


# To calculate the rate of increase of cases from March 1 to March 17
days = 17
NHC_cases = []
rate = 0.3405 # first 17 days no data give, so assume a rate 34.05%
current_value = 1 # assume on March 1, hospitalization cases = 1
for i in range(sums):
    next_value = current_value * math.exp(rate)
    NHC_cases.append(next_value)
    current_value = next_value

NHC_cases = [round(num) for num in NHC_cases]
# NHC_cases


# To generate NHC based on 1-week rate change
count = 0
NHC_cases_1 = []
pp = NHC_cases[16]
# pp

for i in range(0,len(hospitalized)-1):
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
NHC_total_cases = [round(num) for num in NHC_total_cases]
# NHC_total_cases

# To plot

x = np.linspace(0,223,224)
plt.scatter(x,values, label = 'NHC Rate data')
plt.plot(hospitalized, color = 'red', label= 'Observed Data')
plt.axvline(x=20, color ='green', label = 'lockdown')
plt.legend()
# plt.plot(hospitalized)
