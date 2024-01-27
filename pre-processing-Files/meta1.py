
import pandas as pd
import numpy as np
df_val = pd.read_csv(r"C:\My_Data\M2M Data\data\train.csv")

v = df_val['VENDOR']
n = len(pd.unique(df_val['VENDOR']))

n = df_val.nunique(axis=0)




def gen_meta(vendor,scanners_,disease,filed,size):
    
    temp = np.zeros((size))
    
    if vendor=='Philips Medical Systems':
        temp[0,:] = 1
    if vendor=='SIEMENS':
        temp[0,:] = 2
    if vendor=='GE MEDICAL SYSTEMS': 
        temp[0,:] = 3
    
    if scanners_=='Symphony':
        temp[1,:] = -1
    if scanners_=='SIGNA EXCITE':
        temp[1,:] = -2
    if scanners_=='Signa Explorer':
        temp[1,:] = -3
    if scanners_=='SymphonyTim':
        temp[1,:] = -4
    if scanners_=='Avanto Fit':
        temp[1,:] = -5
    if scanners_=='Avanto':
        temp[1,:] = -6
    if scanners_=='Achieva':
        temp[1,:] = -7
    if scanners_=='Signa HDxt':
        temp[1,:] = -8
    if scanners_=='TrioTim':
        temp[1,:] = -9
        
    
    if disease=='NOR':
        temp[2,:] = 0
    if disease=='LV':
        temp[2,:] = 0.5
    if disease=='HCM':
        temp[2,:] = -0.5
    if disease=='ARR':
        temp[2,:] = 0.9
    if disease=='FAIL':
        temp[2,:] = -0.5
    if disease=='CIA':
        temp[2,:] = -0.5
        
    if disease=='1.5':
        temp[3,:] = 1.5
    if disease=='3':
        temp[3,:] = 3


    return temp

size = (4,16,16)
temp = np.zeros((size))



