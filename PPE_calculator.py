%load_ext autotime
import datetime
import pandas as pd
import numpy as np
import glob
from pandas.io.json import json_normalize
pd.set_option('display.max_columns', None)

# Need to update data with API
data_ny = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_ny.csv')
data_az = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_az.csv')
data_ca = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_ca.csv')
data_fl = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_fl.csv')
data_tx = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_tx.csv')
data_nj =  pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_nj.csv')
data_il  = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Covid_tracking_data/daily_il.csv')
# setting at one, so we can read in confirmed case daily from multiple sources
# in this workthough, we used data from https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/

confirmed_case = 1

# one health provider testing 10 patient perday
testing_ratio = 6

# 14% test posittive. (needed for future predictions)
positive_rate = 0.14

inpatient_rate = 0.2
home_care_rate = 1 - inpatient_rate

homecare_rate = 1- inpatient_rate
test_cases = round(confirmed_case / positive_rate)

# 25% of in patient cases are going directly to ICU
# 75% of in patient cases will only need home care
dir_critical_rate = 0.25
support_care_rate = 0.75 

inpatient_care_days = 7
critical_care_days = 17

## patient to lab technician and healthcare workers
patient_tech_ratio = 170


# PPE use per patients per day from in patient to critical care
# room_entries: times / day
nurse_re = 6 
doc_re = 1.2
evs_re = 1
clinic_doc_re = 1
phle_re = 1.5

shift_per_day = 2

# patients to health provider ratio
p2nurse = 2
p2doc = 10
p2evs = 20
p2clinic = 10
p2phle = 10 

#ppe use by technician (ever 170 patiens)
ppe_use_tech = 6 

# means differently through the covid19 patient flow
double = 2

def rate_calc(filename):

    data_states = filename[['dateModified','state','totalTestResultsIncrease','positiveIncrease','negativeIncrease',
                            'hospitalizedCurrently','inIcuCurrently','onVentilatorCurrently','death','recovered']]

    data_states ['date_split'] = pd.to_datetime(data_states ['dateModified'])
    data_states ['date'] = [d.date() for d in data_states['date_split']]
    data_states = data_states.reindex(index = data_states.index[::-1])
    data_states = data_states.reset_index(drop=True)
    data_states = data_states.fillna(0)

    data_states['Test_posrate_%'] = ((data_states['positiveIncrease'] / data_states['totalTestResultsIncrease']) * 100).round(3)
    data_states = data_states.fillna(0)
    return data_states

def rate_calc_list(filename):
    return filename['Test_posrate_%'].tolist()

def date_list(filename):
    return filename['date'].tolist()

def positive_rate_plot(datelist,ratelist,titlename):
    import matplotlib.pyplot as plt

    plt.plot(datelist,ratelist, label = 'Total +ve Tests')
    plt.xlabel('Date: March - July 2020')
    plt.ylabel('% Rate')
    plt.xticks( rotation='vertical')
    plt.title(titlename)
    # show a legend on the plot
    plt.legend()
    # Display a figure
    plt.rcParams['figure.figsize']= [25,10]
    plt.show()
   
def ppes_ed_OB(patients_file):
    patients_file['test_cases'] = patients_file['positiveIncrease']
    # every 30 patients were taked care by 5 healthe providers, see above diagram

    patients_file['healthcareworkers'] = patients_file['test_cases'] / testing_ratio
    patients_file['Droplet_masks'] = patients_file['healthcareworkers']  
    
    patients_file['N95'] =  patients_file['healthcareworkers'] * double # double means two per day
    patients_file['Gloves'] = patients_file['healthcareworkers'] * double # double means left and right hands
    patients_file['Testkits'] =  patients_file['test_cases']
    patients_file['Gowns'] =  patients_file['healthcareworkers']
    
    final1 = patients_file.drop(['healthcareworkers'], axis=1)
    return final1


def ppes_lab_processing(patients_file): 
    """to compute the ppe burn during the lab processing
    Args:
        patients (int): number of total comfirmed cases from community test and ED presentation;   
    Returns:
        total_ppes_lab (dictionary): PPE needs for lab processing
    """
    patients_file['test_cases'] = patients_file['positiveIncrease'] 
     # each technician can run 170 test per day.
    patients_file['technicians'] = patients_file['test_cases'] / patient_tech_ratio 
    patients_file['evs'] = patients_file['test_cases'] / patient_tech_ratio

    patients_file['Droplet_masks'] = ( (patients_file['technicians'] * ppe_use_tech) +  patients_file['evs'] ) 
    patients_file['Gloves'] =  ( (patients_file['technicians'] * ppe_use_tech) +  patients_file['evs'] )  * double
    patients_file['Gowns'] =  patients_file['technicians'] +  patients_file['evs']  


    # total_ppes_lab = dict(Droplet_masks= droplet_masks,
    #                       Gloves= gloves, Gowns=gowns)

    final2 = patients_file.drop(['test_cases', 'technicians','evs','healthcareworkers'], axis=1)
    
    return final2

def ppes_in_patient_care(patients_file):
    """to compute ppe needs during in patient care
    Args:
        patients (int): number of total patients from community test and ED presentation;
        in_patients_days(int): number of days each person will need to be in Patient Care
        
    Returns:
        total_ppes_in_patient (dictionary): PPE needs for in patient care
    """

    # 20% endup being positive and 20% of these in patient care
    # patients_file['in_patient_patients'] = patients_file['positiveIncrease'] * inpatient_rate * support_care_rate

    #use In ICU data since we already have it, we don't have to assume 20%
    
    patients_file['in_patient_patients'] = patients_file['inIcuCurrently']

    # number of health providers each day
    # can't use this to represent the health providers, cause we multiply the room entries
    patients_file['nurses'] = patients_file['in_patient_patients'] / p2nurse * shift_per_day * nurse_re
    patients_file['docs'] = patients_file['in_patient_patients']/p2doc * shift_per_day * doc_re
    patients_file['evss'] = patients_file['in_patient_patients']/p2evs * evs_re
    patients_file['clinicians'] = patients_file['in_patient_patients']/p2doc * shift_per_day * clinic_doc_re
    patients_file['phles'] = patients_file['in_patient_patients']/ p2phle * shift_per_day * phle_re
    
    #total health provider
    patients_file['hps'] = patients_file['nurses'] + patients_file['docs'] + patients_file['clinicians'] + patients_file['phles'] 
    
    
    patients_file['Gloves'] = ((patients_file['hps'] * double) + (patients_file['evss'] * double) ) * inpatient_care_days 
    patients_file['Droplet_masks'] = ( patients_file['hps'] +  (patients_file['evss'] * double) )* inpatient_care_days
    patients_file['Gowns'] = (patients_file['hps'] + (patients_file['evss'] * double) ) * inpatient_care_days 
    
    # goggle can be reused
    patients_file['Goggles'] = (patients_file['hps'] +  (patients_file['evss'] * double) ) * double  
    
    patients_file['BP_cuff'] = patients_file['in_patient_patients'] # bp cuff, ambu bag can be reused

    patients_file['Wiper'] = patients_file['in_patient_patients'] * inpatient_care_days 
    # total_ppes_in_patient = dict(Gloves= gloves, Gowns=gowns, Droplet_masks= droplet_masks,
    #                             Goggles = goggles, BP_cuff = bp_cuff, Wiper = wipers)


    final3 = patients_file.drop(['in_patient_patients','nurses','docs','evss','clinicians','phles','hps'], axis=1)
    
    return final3


def ppes_critical_care(patients_file):
    """compute the ventilators and ppe needs based on the critical care patients
    Args:
        patients (int): number of total patients from community test and ED presentation;
        in_patients_days(int): number of days each person will need to be in Critical Care
        
    Returns:
        total_ppes_criticalcare (dictionary): PPE needs for in critical care
    """
    
    # 40% of in patient patients end up in critical care
    patients_file['critical_care_patients'] =  patients_file['onVentilatorCurrently']
    patients_file['Ventilators'] = patients_file['critical_care_patients']

    # twice room entrences compares to in-patient care
    patients_file['nurses'] = patients_file['critical_care_patients']/p2nurse * shift_per_day * nurse_re * double
    patients_file['docs'] = patients_file['critical_care_patients']/p2doc * shift_per_day * doc_re  * double
    patients_file['evss'] = patients_file['critical_care_patients']/p2evs * evs_re  * double
    patients_file['clinicians'] = patients_file['critical_care_patients']/p2doc * shift_per_day * clinic_doc_re  * double
    patients_file['phles'] = patients_file['critical_care_patients']/ p2phle * shift_per_day * phle_re * double
    
    patients_file['hps'] = patients_file['nurses'] + patients_file['docs'] + patients_file['clinicians'] + patients_file['phles']
   
    
    patients_file['Gloves'] = (patients_file['hps'] + patients_file['evss'] * double) * double *critical_care_days 
    patients_file['Droplet_masks'] = (patients_file['hps'] +  patients_file['evss'] * double )* critical_care_days 
    patients_file['Gowns'] = (patients_file['hps'] +  patients_file['evss'] * double) * critical_care_days 
    patients_file['N95'] = (patients_file['hps'] +  patients_file['evss'] * double )*double * critical_care_days


    
    # goggle can be reused
    patients_file['Goggles'] = (patients_file['hps'] +  (patients_file['evss'] * double) )* double 
    
    patients_file['BP_cuff'] = patients_file['critical_care_patients'] # bp cuff, ambu bag can be reused
    patients_file['Wiper'] = patients_file['critical_care_patients'] * critical_care_days 

    # total_ppes_criticalcare = dict(N95s = n95_needs, Gloves= gloves, Gowns=gowns, 
    #                               Droplet_masks= droplet_masks, Goggles = goggles, 
    #                               BP_cuff = bp_cuff, Wiper = wipers,
    #                               Ventilators = ventilators)
    
    final4 = patients_file.drop(['critical_care_patients','in_patient_patients','nurses','docs','evss','clinicians','phles','hps'], axis=1)
    
    return final4

#update needed
ny_df = rate_calc(data_ny)
az_df = rate_calc(data_az)
ca_df = rate_calc(data_ca)
fl_df = rate_calc(data_fl)
tx_df = rate_calc(data_tx)
nj_df = rate_calc(data_nj)
il_df = rate_calc(data_il)

ny_pov_ratelist = rate_calc_list(ny_df)
az_pov_ratelist = rate_calc_list(az_df)
ca_pov_ratelist = rate_calc_list(ca_df)
fl_pov_ratelist = rate_calc_list(fl_df)
tx_pov_ratelist = rate_calc_list(tx_df)
nj_pov_ratelist = rate_calc_list(nj_df)
il_pov_ratelist = rate_calc_list(il_df)

date_list = date_list(ny_df)

ny_ppe_req = ppes_ed_OB(ny_df)
# ny_ppe_req
ny_ppe_req_criticalcare = ppes_lab_processing(ny_df)
# ny_ppe_req_lab
ny_ppe_req_inpatient = ppes_in_patient_care(ny_df)
# ny_ppe_req_inpatient
ny_ppe_req_criticalcare = ppes_critical_care(ny_df)
# ny_ppe_req_criticalcare

# for every confirmed convid19 case from home in patient and critical cares 
ny_df_final = ny_df
ny_df_final['Total_Testkits'] = ny_df_final['positiveIncrease']
ny_df_final['Total_N95s'] = ny_ppe_req['N95'] + ny_ppe_req_criticalcare['N95'] 
ny_df_final['Total_Droplet_masks'] =  ny_ppe_req ['Droplet_masks'] + ny_ppe_req_criticalcare ['Droplet_masks'] + ny_ppe_req_inpatient['Droplet_masks'] + ny_ppe_req_criticalcare ['Droplet_masks']
ny_df_final['Total_Gloves'] = ny_ppe_req ['Gloves']+ ny_ppe_req_criticalcare ['Gloves'] + ny_ppe_req_inpatient['Gloves'] + ny_ppe_req_criticalcare ['Gloves']  
ny_df_final['Total_Gowns'] = ny_ppe_req ['Gowns']+ ny_ppe_req_criticalcare ['Gowns'] + ny_ppe_req_inpatient['Gowns'] + ny_ppe_req_criticalcare ['Gowns'] 
ny_df_final['Total_Goggles'] = ny_ppe_req_inpatient['Goggles'] + ny_ppe_req_criticalcare ['Goggles']
ny_df_final['Total_BP_cuffs'] = ny_ppe_req_inpatient['BP_cuff'] + ny_ppe_req_criticalcare ['BP_cuff']
ny_df_final['Total_Wipers'] = ny_ppe_req_inpatient['Wiper'] + ny_ppe_req_criticalcare ['Wiper']
ny_df_final['Total_Ventilators'] = ny_ppe_req_criticalcare ['Ventilators']

ny_df_final['Total_PPE'] = ny_df_final['Total_Testkits'] + ny_df_final['Total_N95s'] + ny_df_final['Total_Droplet_masks'] + ny_df_final['Total_Gloves'] + ny_df_final['Total_Gowns'] + \
                            ny_df_final['Total_Goggles'] + ny_df_final['Total_BP_cuffs'] + ny_df_final['Total_Wipers'] + ny_df_final['Total_Ventilators'] 
ny_df_final.round(2)

ny_df_final = ny_df_final[['dateModified', 'state', 'totalTestResultsIncrease', 'positiveIncrease', 'negativeIncrease', 'hospitalizedCurrently','onVentilatorCurrently','inIcuCurrently'
                            'death', 'recovered', 'date_split', 'date', 'Test_posrate_%','Total_Testkits','Total_N95s','Total_Droplet_masks',
                            'Total_Gloves','Total_Gowns','Total_Goggles','Total_BP_cuffs','Total_Wipers','Total_Ventilators','Total_PPE']]

ny_df_final.to_csv('/Users/pritishsadiga/Desktop/Twitter/PPE_github/ny_ppe_demand2.csv')

#Code for normalising dat
from sklearn import preprocessing
def normaliser(filename):
    
    file_float = filename.values.astype(float)
   # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    file_scaled = min_max_scaler.fit_transform(file_float)
    

    # Run the normalizer on the dataframe
    file_normalized = pd.DataFrame(file_scaled)
    file_normalized = file_normalized.fillna(0)
    
    file_norm_list = file_normalized.iloc[:,0].tolist()
    
    return file_norm_list 
    
ny_ppe_demand = normaliser(ny_df_final[['Total_PPE']])
ny_pov_ratelist = normaliser(ny_df[['Test_posrate_%']])

a = pd.read_csv('/Users/pritishsadiga/Desktop/Twitter/Google_Trends/PPE_shortage_US/multiTimeline_NY.csv')
listt = normaliser(a[['shortage']])

import matplotlib.pyplot as plt

plt.plot(date_list,ny_pov_ratelist, label = 'Total +ve confirmed cases')
plt.plot(date_list,ny_ppe_demand, label = 'NY-PPE Demand')
plt.plot(date_list,listt, label = 'Google Trends')
plt.xlabel('Date: March - July 2020')
plt.ylabel('Rate')
plt.xticks(rotation='vertical')
plt.title('New York')
# show a legend on the plot
plt.legend()
# Display a figure
plt.rcParams['figure.figsize']= [25,10]
plt.savefig('/Users/pritishsadiga/Desktop/graph.png')

plt.show()

'''
ny_testkits = normaliser(ny_df_final[['Total_Testkits']])
ny_N95 = normaliser(ny_df_final[['Total_N95s']])
ny_Dropletmasks = normaliser(ny_df_final[['Total_Droplet_masks']])
ny_gloves = normaliser(ny_df_final[['Total_Gloves']])
ny_goggles = normaliser(ny_df_final[['Total_Goggles']])
ny_bpcuff = normaliser(ny_df_final[['Total_BP_cuffs']])
ny_wipers = normaliser(ny_df_final[['Total_Wipers']]) 
ny_ventilators = normaliser(ny_df_final[['Total_Ventilators']])
import matplotlib.pyplot as plt

# plt.plot(date_list,ny_pov_ratelist, label = 'Total +ve confirmed cases')
plt.plot(date_list,ny_testkits, label = 'NY-Testkits Demand')
plt.plot(date_list,ny_N95, label = 'NY-N95 Demand')
plt.plot(date_list,ny_Dropletmasks, label = 'NY-Droplet Masks Demand')
plt.plot(date_list,ny_gloves, label = 'NY-Gloves Demand')
plt.plot(date_list,ny_goggles, label = 'NY-Goggles Demand')
plt.plot(date_list,ny_bpcuff, label = 'NY-BP Cuff Demand')
plt.plot(date_list,ny_wipers, label = 'NY-Wipers Demand')
plt.plot(date_list,ny_ventilators, label = 'NY-Ventilators Demand')

plt.xlabel('Date: March - July 2020')
plt.ylabel('Rate')
plt.xticks(rotation='vertical')
plt.title('New York')
# show a legend on the plot
plt.legend()
# Display a figure
plt.rcParams['figure.figsize']= [25,10]

plt.show() 
