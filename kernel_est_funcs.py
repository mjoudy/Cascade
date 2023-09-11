# this script consists of functions to simulate calcium signals from spike trains,
#smooth them using savitzky-golay and 4 different outlier removal methods in order to 
#for a line to scatter plot of signal-derivative of signal.

import numpy as np
#import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig
import seaborn as sns

from scipy.interpolate import interp1d
from sklearn.linear_model import RANSACRegressor

#plt.style.use('ggplot')
plt.style.use('seaborn')

#sns.set_style('white')
 


def sim_calcium(spikes, tau=100, neuron_id=500):

    #N = np.shape(spikes)[0]
    wup_time = 1000
    spikes = spikes[neuron_id, wup_time:]
    sim_dur = np.shape(spikes)[0]

    noise_intra = np.random.normal(0, 0.01, sim_dur)
    spikes_noisy = spikes + noise_intra

    calcium = np.zeros(sim_dur)
    calcium_nsp = np.zeros(sim_dur)
    dt = 1
    const_A = np.exp((-1/tau)*dt)

    calcium[0] = spikes[0]
    calcium_nsp[0] = spikes[0]

    for t in range(1, sim_dur):
        calcium[t] = const_A*calcium[t-1] + spikes[t]


    for t in range(1, sim_dur):
        calcium_nsp[t] = const_A*calcium_nsp[t-1] + spikes_noisy[t]

    noise_recording = np.random.normal(0,1, sim_dur)
    calcium_noisy = calcium + noise_recording
    calcium_nsp_noisy = calcium_nsp + noise_recording

    #return calcium, calcium_noisy, calcium_nsp, calcium_nsp_noisy
    return calcium_nsp_noisy


def smoothed_signals(signal, win_len, neg = False, do_plots=False):
    smooth_cal = sig.savgol_filter(signal, window_length=win_len, deriv=0, delta=1., polyorder=3)
    smooth_deriv = sig.savgol_filter(signal, window_length=win_len, deriv=1, delta=1., polyorder=3)

    negetive_cal = smooth_cal[smooth_deriv < 0]
    negetive_deriv = smooth_deriv[smooth_deriv < 0]

    if do_plots==True:
        fig, (ax1, ax2) = plt.subplots(2,1,figsize=(20,8))
        ax1.plot(smooth_cal)
        ax1.plot(30*smooth_deriv)
        ax2.plot(negetive_cal)
        ax2.plot(30*negetive_deriv)

    if neg == True:
        return negetive_cal, negetive_deriv
    else:
        return smooth_cal, smooth_deriv
    
#this function is defined just for cascade project
def upsample(times, dff, spikes, new_rate, intp_method = 'cubic', do_plot=False):

    time_start = times[0]
    shifted_time = times - time_start
    intpld_signal_func = interp1d(shifted_time, dff, kind=intp_method)
    evenly_spaced_time = np.linspace(shifted_time[0], shifted_time[-1], int((shifted_time[-1])*new_rate))
    upsampled_signal = intpld_signal_func(evenly_spaced_time)
    time_shift = np.zeros(int(times[0]*new_rate))
    upsampled_signal = np.concatenate((time_shift, upsampled_signal), axis=0)
    upsampled_spikes = new_rate*spikes
    
    if do_plot:
        plt.figure(figsize=(20,5))
        plt.plot(upsampled_signal)
        max_dff = np.max(dff[4:])
        min_dff = np.min(dff[4:])
        plt.eventplot(upsampled_spikes, lineoffsets=min_dff - max_dff/20, linelengths=max_dff/20, color='k')

    return upsampled_signal, upsampled_spikes

'''
def cut_spikes(spikes, signal, deriv, win_len=5):
    
    bool_check = all(element==0 or element==1 for element in spikes)

    if bool_check:
        spikes = np.where(spikes)[0]
    else:
        spikes = spikes.astype(int)

    remove_index = []
    for i in spikes:
        remove_index.append(np.arange(i-win_len, i+win_len))
        #add a line to include cut spikes
    remove_index = np.array(remove_index)
    remove_index = remove_index.flatten()
    remove_index = remove_index[remove_index>0]

    signal = np.delete(signal, remove_index)
    deriv = np.delete(deriv, remove_index)

    return signal, deriv

'''

def cut_spikes(spikes, signal, deriv, win_len=5):
    
    bool_check = all(element==0 or element==1 for element in spikes)

    if bool_check:
        event_spikes = np.where(spikes)[0]
    else:
        event_spikes = spikes.astype(int)

    remove_index = []
    for i in event_spikes:
        remove_index.append(np.arange(i-win_len, i+win_len))
        #add a line to include cut spikes
    
    remove_index = np.array(remove_index)
    remove_index = remove_index.flatten()
    remove_index = remove_index[remove_index>0]
    remove_index = remove_index[remove_index<len(signal)]

    signal = np.delete(signal, remove_index)
    deriv = np.delete(deriv, remove_index)

    return signal, deriv


def scatter_all(signal, win_len):
    smooth_cal = sig.savgol_filter(signal, window_length=win_len, deriv=0, delta=1., polyorder=3)
    smooth_deriv = sig.savgol_filter(signal, window_length=win_len, deriv=1, delta=1., polyorder=3)

    fig, ax = plt.subplots()
    ax.scatter(smooth_cal, smooth_deriv, marker='.', s=5)
    ax.set_xlabel('Calcium Signal')
    ax.set_ylabel('Derivative of Calcium Signal')


def pure_fit(signal, deriv, do_plot=False):

    b_pure_fit, a_pure_fit = np.polyfit(signal, deriv, deg=1)

    if do_plot==True:
        fig, ax = plt.subplots()
        ax.scatter(signal, deriv, marker='.', s=5)
        ax.plot(signal, a_pure_fit+b_pure_fit*signal, color='k')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')
    
    return b_pure_fit


def iqr_outlier(signal, deriv, threshold=1.5, percentile_start=25, percentile_end=75, do_plot=False):
    
    x = np.array(signal)
    y = np.array(deriv)

    residuals = y - np.polyval(np.polyfit(x, y, 1), x)
    quartile_1, quartile_3 = np.percentile(residuals, [percentile_start, percentile_end])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (threshold * iqr)
    upper_bound = quartile_3 + (threshold * iqr)
    mask = (residuals >= lower_bound) & (residuals <= upper_bound)

    inlier_x = x[mask]
    inlier_y = y[mask]

    b_iqr, a_iqr = np.polyfit(x[mask], y[mask], deg=1)
    
    if do_plot==True:
        i_mask = np.logical_not(mask)
        outlier_x = x[i_mask]
        outlier_y = y[i_mask]
        fig, ax = plt.subplots()
        ax.scatter(inlier_x, inlier_y, marker='.', s=5)
        ax.plot(inlier_x, a_iqr+b_iqr*inlier_x, color='k')
        ax.scatter(outlier_x, outlier_y, marker='.', s=5, color='red')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    return b_iqr


def ransac_outlier(signal, deriv, do_plot=False):

    ransac = RANSACRegressor()
    x = np.array(signal).reshape(-1, 1)
    y = np.array(deriv)
    ransac.fit(x, y)

    if do_plot==True:
        mask = ransac.inlier_mask_
        i_mask = np.logical_not(mask)
        line_x = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        line_y = ransac.predict(line_x)

        fig, ax = plt.subplots()
        plt.scatter(x[mask], y[mask], label='Inliers', marker='o', s=5)
        plt.scatter(x[i_mask], y[i_mask], label='Outliers', color='red', marker='o', s=5)
        plt.plot(line_x, line_y, color='green', label='Robust Regression Line')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    
    return ransac.estimator_.coef_[0]


def zscore_outlier(signal, deriv, threshold=2, do_plot=False):

    x = np.array(signal)
    y = np.array(deriv)
    z_scores_x = (signal - np.mean(signal)) / np.std(signal)
    z_scores_y = (deriv - np.mean(deriv)) / np.std(deriv)
    mask = np.abs(z_scores_x) < threshold
    mask &= np.abs(z_scores_y) < threshold

    b_zscore, a_zscore = np.polyfit(x[mask], y[mask], deg=1)

    if do_plot==True:
        i_mask=np.logical_not(mask)
        fig, ax = plt.subplots()
        plt.scatter(x[mask], y[mask], marker='.', s=5)
        plt.scatter(x[i_mask], y[i_mask], marker='.', s=5,color='red')
        plt.plot(x[mask], a_zscore+b_zscore*x[mask], color='k')
        ax.set_xlabel('Calcium Signal')
        ax.set_ylabel('Derivative of Calcium Signal')

    return b_zscore



