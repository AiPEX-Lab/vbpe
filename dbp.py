import numpy as np 
import cv2
import os 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import random
import statistics
from vessel_length import *
from scipy.signal import butter, lfilter, lfilter_zi
import gc

def dbp_butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def dbp_butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = dbp_butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def dbp_butter_bandpass_filter_zi(data, lowcut, highcut, fs, order=5):
    b, a = dbp_butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a)
    y,zo = lfilter(b, a, data, zi=zi*data[0])
    return y

def dbp_lag_finder(y1, y2, sr):
    n = len(y1)

    corr = signal.correlate(y2, y1, mode='same') / np.sqrt(signal.correlate(y1, y1, mode='same')[int(n/2)] * signal.correlate(y2, y2, mode='same')[int(n/2)])

    delay_arr = np.linspace(-0.5*n/sr, 0.5*n/sr, n)
    delay = delay_arr[np.argmax(corr)]
    return abs(delay)

def dbp_peak_detector(sig1, sig2):
    p1 = signal.argrelextrema(sig1, np.less)
    p2 = signal.argrelextrema(sig2, np.less)
    p1 = np.expand_dims(np.asarray(p1[0]), axis=1)
    p2 = np.expand_dims(np.asarray(p2[0]), axis=1)
    min_len = min(len(p1), len(p2)) - 1
    diff = abs(p1[0:min_len, :] - p2[0:min_len, :])
    tdelay = np.mean(diff)
    return tdelay

def dbp_vessel_radius(height_person_cm, age, bmi):
    return -0.258 + 0.029*height_person_cm + 0.006*age + 0.036*bmi

def dbp_vessel_thickness(height_person_cm, age, bmi):
    return 0.25 + 0.005*age + 0.005*bmi
    
def dbp_calculator(lag, frame_rate, image_path, height_cm, age, bmi):
    # vessel_length = 0.01*0.2
    vessel_length = 0.01*get_vessel_length(image_path, height_cm)
    # print('Vessel Length', vessel_length)
    beta = 8.5 # Original
    rho = 1060 # Original
    lagf = abs(lag)/frame_rate
    # alpha = 0.015 # Original
    alpha = 0.017
    # r = 0.005 # Original
    r = 0.001*0.5*dbp_vessel_radius(height_cm, age, bmi)
    # print('Vessel Radius', r)
    # h = 0.001 # Original
    h = 0.001*dbp_vessel_thickness(height_cm, age, bmi)
    # print('Thickness', h)
    # E_0 = 263000 # Original
    E_0 = 1005
    bp = -0.12*((-2/alpha)*np.log(lagf) + (1/alpha)*np.log((2*r*rho*(vessel_length**2))/(h*E_0)))
    return bp

def main_dbp(ppg_path_face, ppg_path_hand, image_path, demographic_path):
    csv_face_names = sorted([file for file in os.listdir(ppg_path_face) if file.endswith('.csv')])
    csv_hand_names = sorted([file for file in os.listdir(ppg_path_hand) if file.endswith('.csv')])
    image_names = sorted([file for file in os.listdir(image_path) if file.endswith('.jpg')])
    demographic_data = pd.read_csv(demographic_path)
    height_cm = demographic_data['Height'].tolist()
    age = demographic_data['Age'].tolist()
    bmi = demographic_data['BMI'].tolist()
    
    bp_values = []
    lowcut = 0.25
    highcut = 15
    sRate = 60

    for csv_face, csv_hand, image, h, a, b in zip(csv_face_names, csv_hand_names, image_names, height_cm, age, bmi):
        d1 = pd.read_csv(os.path.join(ppg_path_face, csv_face)).squeeze()
        d2 = pd.read_csv(os.path.join(ppg_path_hand, csv_hand)).squeeze()
        d1 = -1*d1
        d2 = -1*d2
        image_name_path = os.path.join(image_path, image)
        fps = 60

        d1butter = dbp_butter_bandpass_filter_zi(d1, lowcut, highcut, sRate, order=2)
        d2butter = dbp_butter_bandpass_filter_zi(d2, lowcut, highcut, sRate, order=2)
        time_delay = dbp_peak_detector(d1butter, d2butter)
        bp_predicted = dbp_calculator(time_delay, fps, image_name_path, h, a, b)
        bp_values.append(bp_predicted)

        del d1, d2, d1butter, d2butter
        gc.collect()
    print(csv_face_names, csv_hand_names)
    print('Diastolic Blood pressure:', bp_values)
    np.savetxt('DBP_new.csv', bp_values, header='DBP')

    return bp_values

