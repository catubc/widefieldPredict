import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import pandas as pd
import os
import shutil
import cv2
import scipy.io as sio
import scipy.signal

from Specgram.Specgram import Specgram

import glob2

from numba import jit
from sklearn.svm import SVC # "Support vector classifier"



# functions
def plot_median_sem_over_single_trials(area_ids, trial_courses):
    ymin, ymax = -10, 10
    for k in range(area_ids.shape[0]):
        ax=plt.subplot(6,6,k+1)
        temp = trial_courses[:,k]
        median = np.nanmedian(temp, axis=0)
        print ("Area: ", area_ids[k])
        # compute STD and SEM
        std = np.nanstd(temp,axis=0)
        sem = std/float(trial_courses.shape[0])
        #print ("sem: ", sem.shape, " , mean: ", median.shape)
        plt.plot(median,c='blue')

        # plot individual trials
        #plt.plot(temp.T, c='black',alpha=.1)

    #     plt.fill_between(np.arange(mean.shape[0]), mean-std, mean+std,
    #                    alpha=0.2, facecolor='#089FFF',
    #                    linewidth=4, antialiased=True)

        plt.plot([0, trial_courses.shape[2]],[0,0], 'r--',c='black')
        plt.plot([trial_courses.shape[2]//2, trial_courses.shape[2]//2],
                [ymin, ymax], 'r--',c='black')

#         if area_ids[k]==100:
#             print ("Median: ", median, temp)
            # mean[mean == -np.inf] = 0
       # mean[mean == np.inf] = 0

        plt.ylim(-10, 10)
        #plt.fill_between(mean, mean-sem, mean+sem,color='grey')
        plt.plot(median+sem, color='red')
        plt.plot(median-sem, color='red')
        #idx = np.where(areanames[:,0]==area_ids[k])[0]
        #print (allen_ids[:10], area_ids[k])
        #idx = np.where(allen_ids-1==area_ids[k])[0]

        #print (allen_abbreviations[idx])
        #plt.title("Area: "+ allen_abbreviations[idx][0]+ ", "+str(area_ids[k]),
        #plt.title("Area: "+ allen_abbreviations[idx][0]+ ", "+str(area_ids[k]),
        plt.title("Area: "+ str(area_ids[k]),
                  fontsize=12, pad=.9)

        if k < (area_ids.shape[0]-1):
            plt.xticks([])

        if k == 0:
            plt.ylabel("DF/F %",fontsize=15)

    plt.suptitle("All significant areas (10pixels+): medians over # trials: "+
                 str(trial_courses.shape[0]), fontsize=16)

    plt.show()


def sum_pixels_in_registered_mask(data, maskwarp):

    print (" # of trials: ", data.shape[0])

    areas = np.unique(maskwarp) 
    print (" # of areas: ", areas.shape)
    
    # work in 1D vectors easier to mask
    maskwarp1D = maskwarp.reshape(-1)

    trial_courses = []
    area_ids = []
    for k in range(data.shape[0]):
        if k%10==0:
            print ("computing trial: ", k)
        time_courses_local = []

        # convert to 1D vector to mask faster
        data1D = np.float32(data[k].reshape(181,-1))
        for id_ in areas:
            idx = np.where(maskwarp1D==id_)[0]

            # only keep areas that have at least 10 pixels
            if idx.shape[0]>10:
                #print ("Area: ", id_)
                area_ids.append(id_)#print ("Areas: ", id_)
                #print (data1D[:,idx].shape)

                temp = data1D[:,idx]

                if False:
                    # compute DFF
                    F0 = np.nanmean(temp,axis=0)
                    dFF = (data1D[:,idx]-F0)/F0
                else:
                    # skip dFF computation;
                    dFF = temp
                
                # save average of all pixesl post DFF
                time_courses_local.append(np.nanmean(dFF, axis=1))

            #all_times.append(time_courses_local)
        trial_courses.append(time_courses_local)

    area_ids = np.int32(np.unique(area_ids))
    trial_courses = np.float32(trial_courses)
    print ("# trials, # areas, # times: ", trial_courses.shape)
    print ("area ids: ", area_ids.shape)

    return area_ids, trial_courses


def make_movie(data):
    
    from matplotlib import animation
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    data = np.float32(data)
    print (data.shape)
    # data_trial = data[trial]

    # F0 = np.mean(data_trial[:30],axis=0)
    # print (F0.shape)

    # dFF = (data.mean(0) - F0)/F0
    # print ("dFF: ", dFF.shape)

    #dFF = data.mean(0)

    trial = 0
    print (np.nanmin(data), np.nanmax(data))
    n_frames = data.shape[0]
    print ("n_frames: ", n_frames)
    
    im = ax.imshow(data[data.shape[0]//2], vmin=-0.1, vmax=0.1, cmap='viridis')
    #im.set_clim([0,1])
    ax.set_title("")
    fig.set_size_inches([5,5])
   
    def update_img(n):
        #tmp = rand(300,300)
        
        print (n)
        ax.set_title(str(n))
        im.set_data(data[n])
        return im

    #legend(loc=0)
    ani = animation.FuncAnimation(fig, update_img,n_frames,interval=30)
    writer = animation.writers['ffmpeg'](fps=30)

    ani.save('/home/cat/video.mp4',writer=writer)
    plt.close()

def make_training_sets_multiple_tests_window(time, 
                                      trial_courses_fixed, trial_courses_fixed_ids,
                                      trial_courses_random_fixed, trial_courses_random_ids):
    
    # combine good trials with random trials for training sets:
    
    #time = 0
    good_trials = trial_courses_fixed[trial_courses_fixed_ids, :,time:time+30].reshape(trial_courses_fixed_ids.shape[0], -1)
    #print ("good trials: ", good_trials.shape)
    temp = np.arange(trial_courses_fixed.shape[0])
    idx = np.delete(temp,trial_courses_fixed_ids)
    test_trials = trial_courses_fixed[idx, :,time:time+30].reshape(idx.shape[0], -1)

    #print ("test_trials: ", test_trials.shape)
    
    random_trials = trial_courses_random_fixed[trial_courses_random_ids, :,time:time+30].reshape(trial_courses_random_ids.shape[0], -1)
    temp = np.arange(trial_courses_random.shape[0])
    idx = np.delete(temp,trial_courses_random_ids)
    test_trials_random = trial_courses_random_fixed[idx, :,time:time+30].reshape(idx.shape[0], -1)
    #print ("test_trials_random: ", test_trials_random.shape)

    # make labels
    y = np.zeros(good_trials.shape[0]+random_trials.shape[0],'int32')
    y[:good_trials.shape[0]]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    print ("done time: ", time)
    return X, y, test_trials, test_trials_random


def make_training_sets_multiple_tests(time, 
                                      trial_courses_fixed, trial_courses_fixed_ids,
                                      trial_courses_random_fixed, trial_courses_random_ids):
    
    # combine good trials with random trials for training sets:
    
    #time = 0
    good_trials = trial_courses_fixed[trial_courses_fixed_ids, :,time].reshape(trial_courses_fixed_ids.shape[0], -1)
    #print ("good trials: ", good_trials.shape)
    temp = np.arange(trial_courses_fixed.shape[0])
    idx = np.delete(temp,trial_courses_fixed_ids)
    test_trials = trial_courses_fixed[idx, :,time].reshape(idx.shape[0], -1)

    #print ("test_trials: ", test_trials.shape)
    
    random_trials = trial_courses_random_fixed[trial_courses_random_ids, :,time].reshape(trial_courses_random_ids.shape[0], -1)
    temp = np.arange(trial_courses_random.shape[0])
    idx = np.delete(temp,trial_courses_random_ids)
    test_trials_random = trial_courses_random_fixed[idx, :,time].reshape(idx.shape[0], -1)
    #print ("test_trials_random: ", test_trials_random.shape)

    # make labels
    y = np.zeros(good_trials.shape[0]+random_trials.shape[0],'int32')
    y[:good_trials.shape[0]]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    print ("done time: ", time)
    return X, y, test_trials, test_trials_random



def make_training_sets(time, trial_courses_fixed, trial_courses_random_fixed):
    # combine good trials with random trials for training sets:
    
    #time = 0
    good_trials = trial_courses_fixed[:50, :,time].reshape(50, -1)
    #print ("good trials: ", good_trials.shape)
    test_trials = trial_courses_fixed[50:, :,time].reshape(13, -1)

    random_trials = trial_courses_random_fixed[:50, :,time].reshape(50, -1)
    #print ("random_trials: ", random_trials.shape)
    test_trials_random = trial_courses_random_fixed[50:, :,time].reshape(50, -1)

    # make labels
    y = np.zeros(100,'int32')
    y[:50]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    print ("done time: ", time)
    return X, y, test_trials, test_trials_random

    
def make_training_sets_multi_times(times, trial_courses_fixed, trial_courses_random_fixed):
    # combine good trials with random trials for training sets:
    
    #time = 0
    good_trials = trial_courses_fixed[:50, :,times[0]:times[1]].reshape(50, -1)
    #print ("good trials: ", good_trials.shape)
    test_trials = trial_courses_fixed[50:, :,times[0]:times[1]].reshape(13, -1)

    random_trials = trial_courses_random_fixed[:50, :,times[0]:times[1]].reshape(50, -1)
    #print ("random_trials: ", random_trials.shape)
    test_trials_random = trial_courses_random_fixed[50:, :,times[0]:times[1]].reshape(50, -1)

    # make labels
    y = np.zeros(100,'int32')
    y[:50]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    print ("done time: ", time)
    return X, y, test_trials, test_trials_random

def make_training_sets_multi_times_multi_areas(times, 
                                               area_id,
                                               trial_courses_fixed, 
                                               trial_courses_random_fixed):
    # combine good trials with random trials for training sets:
    
    #time = 0
    good_trials = trial_courses_fixed[:50, area_id,times[0]:times[1]].reshape(50, -1)
    #print ("good trials: ", good_trials.shape)
    test_trials = trial_courses_fixed[50:, area_id,times[0]:times[1]].reshape(13, -1)

    random_trials = trial_courses_random_fixed[:50, area_id,times[0]:times[1]].reshape(50, -1)
    #print ("random_trials: ", random_trials.shape)
    test_trials_random = trial_courses_random_fixed[50:, area_id,times[0]:times[1]].reshape(50, -1)

    # make labels
    y = np.zeros(100,'int32')
    y[:50]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    #print ("done time: ", time)
    return X, y, test_trials, test_trials_random


def parallel_svm_multiple_tests(time, 
                                trial_courses_fixed, trial_courses_fixed_ids,
                                trial_courses_random_fixed, trial_courses_random_ids):

    res1 = []
    res2 = []
    for k in range(len(trial_courses_fixed_ids)):
#         X, y, test_trials, test_trials_random = make_training_sets_multiple_tests(time, 
#                                                                    trial_courses_fixed, trial_courses_fixed_ids[k],
#                                                                    trial_courses_random_fixed, trial_courses_random_ids[k])
        X, y, test_trials, test_trials_random = make_training_sets_multiple_tests_window(time, 
                                                                   trial_courses_fixed, trial_courses_fixed_ids[k],
                                                                   trial_courses_random_fixed, trial_courses_random_ids[k])

        #print (" X: ", X.shape, X[:5])
        print (" y: ", y.shape, y)
        
        model = SVC(kernel='linear', C=1)
        model.fit(X, y)

        #test_trials_rewarded = trial_courses_fixed[50:, :,time].reshape(50, -1)
        #model = grid.best_estimator_
        yfit = model.predict(test_trials)
        print ("predict test trial: ", yfit)
        res1.append(np.sum(yfit)/float(yfit.shape[0]))
        #real_data.append(res1)

        yfit = model.predict(test_trials_random)
        res2.append(np.sum(yfit)/float(yfit.shape[0]))
        #random_data.append(res2)
    
    return (res1, res2)

def parallel_svm_multi_time(times, trial_courses_fixed, trial_courses_random_fixed):
#for time in times:
    print ("times ", times)
    X, y, test_trials, test_trials_random = make_training_sets_multi_times(times, 
                                                                           trial_courses_fixed, 
                                                                           trial_courses_random_fixed)
    
    model = SVC(kernel='linear', C=2)
    model.fit(X, y)

    # test_trials_rewarded = trial_courses_fixed[50:, :,time].reshape(50, -1)
    # model = grid.best_estimator_
    yfit = model.predict(test_trials)
    res1 = np.sum(yfit)/float(yfit.shape[0])
    real_data.append(res1)
    
    yfit = model.predict(test_trials_random)
    res2 = np.sum(yfit)/float(yfit.shape[0])
    random_data.append(res2)
    
    print (res1, res2)
    return (res1, res2)

def parallel_svm_multi_time_multi_area(times, 
                                       areas,
                                       trial_courses_fixed, 
                                       trial_courses_random_fixed):
    print ("times ", times)
    
    res1_array = []
    res2_array = []
    
    for area_id in areas:
        X, y, test_trials, test_trials_random = make_training_sets_multi_times_multi_areas(
                                                                               times, 
                                                                               area_id,
                                                                               trial_courses_fixed, 
                                                                               trial_courses_random_fixed)

        model = SVC(kernel='linear', C=2)
        model.fit(X, y)

        # test_trials_rewarded = trial_courses_fixed[50:, :,time].reshape(50, -1)
        # model = grid.best_estimator_
        yfit = model.predict(test_trials)
        res1 = np.sum(yfit)/float(yfit.shape[0])
        res1_array.append(res1)

        yfit = model.predict(test_trials_random)
        res2 = np.sum(yfit)/float(yfit.shape[0])
        res2_array.append(res2)

    print ("done times: ", times)
    return (res1_array, res2_array)


# find sudden movements in time series
def find_starts(feature):
    plotting = False
    

    times = np.arange(feature.shape[0])/15. #- 0.5
    #print (times)
    fs=15
    P, extent = Specgram(data = feature, sampfreq = fs,
                         p0=-60, f0=0.5, f1=fs,
                         width=0.25, tres = 0.125)


    data1 = P[0]-np.median(P[0])
    data2 = np.abs(feature)

    # only need to find this shift once!
    rolling_data = []
    for k in range(-20,20,1):
        rolling_data.append((data1*np.roll(data2[::2][:-1],k)).sum())

    #shift = np.argmax(rolling_data)-20
    shift = -1
    print ("HARDCODED feature shift: ", shift)
    
    #plot maximum for the roll shift;
    if plotting:
        fig = plt.figure()
        ax = plt.subplot(2,1,1)
        plt.plot(rolling_data)

    P_med = P[0]-np.median(P[0])
    starts = []
    for k in range(1,P_med.shape[0],1):
        if (P_med[k]>P_med[k-1]) and (P_med[k-1]==0):
            starts.append(k)
    starts = np.array(starts)/15.*2

    
    # plot feature traces, power-spectrogram peaks, argrelmax peaks and initiation peaks
    if plotting:
        locs = np.array(scipy.signal.argrelmax(P_med)).squeeze()/15.*2
        ax=plt.subplot(2,1,2)
        plt.plot(times+shift/15., np.abs(feature))
        plt.plot(times[::2][:-1],P_med, c='red')
        plt.scatter(locs, P_med[np.int32(locs*15/2.)], c='green')
        plt.scatter(starts, P_med[np.int32(starts*15/2)]*1.1, c='orange')
        plt.show()

    return starts

def visualize_lever_vs_DLClever(starts_arrays, abscodes,abstimes):

    labels = ['left_paw','right_paw','nose','lever','right_ear','jaw','tongue']

    # convert abspositiosn to integers
    vals = []
    for k in range(abscodes.shape[0]):
        vals.append(np.int(abscodes[k].decode()))
    vals=np.array(vals)

    idx04 = np.where(vals==4)[0]
    idx02 = np.where(vals==2)[0]

    fig=plt.figure()
    ax=plt.subplot(111)
    
    # plot lever starts
    for k in range(len(starts_arrays)):
        if k==3:
            ax.scatter(starts_arrays[k], starts_arrays[k]*0+k, label=labels[k])
    
    # plot lever codes
    ax.scatter(abstimes[idx04], idx04*0+8, c='darkblue', label='04 codes')
    ax.scatter(abstimes[idx02], idx02*0+9, c='darkgreen', label='02 codes')

    # metadata
    ax.legend(title="Behaviours initiated by", fontsize=15)
    plt.xlabel("Time (sec)",fontsize=20)
    plt.yticks([])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.title("DLC traced behaviour initiations by body part ", fontsize=20)

    # compute crosscorrelogram between lever starts and 
    shifts = np.arange(-5.0, 5.0, 0.03)
    arraysum = np.zeros(1400*1000,'int16')
    
    # select only the first entry in a series
    starts_04 = []
    starts_04.append(abstimes[idx04][0])
    for k in range(1, abstimes[idx04].shape[0], 1):
        if abstimes[idx04][k]-abstimes[idx04][k-1]>0.100:
            starts_04.append(abstimes[idx04][k])
    starts_04 = np.array(starts_04)

    starts_02 = []
    starts_02.append(abstimes[idx02][0])
    for k in range(1, abstimes[idx02].shape[0], 1):
        if abstimes[idx02][k]-abstimes[idx02][k-1]>0.100:
            starts_02.append(abstimes[idx02][k])
    starts_02 = np.array(starts_02)
    
    #
    feature = 3 # select lever feature
    results = []
    for shift in shifts:
        arraysum*=0
        
        #print
        arraysum[np.int32(starts_arrays[feature])]+=1
        arraysum[np.int32(starts_04+shift)]+=1
        arraysum[np.int32(starts_02+shift)]+=1
        #arraysum[np.int32(abstimes[idx02]+shift)]+=1

        idx = np.where(arraysum>1)[0]
        results.append(idx.shape[0])
    
    fig=plt.figure()
    ax=plt.subplot(111)
    plt.plot(shifts, results)
    plt.title("Empirical crosscorrelogram between DLC lever starts and lever traces")
    plt.show()
    
    idx = np.argmax(results)
    best_shift = shifts[idx]
    
    print (" best shift: ", best_shift, "sec")
    
    return best_shift
    
def visualize_lever_vs_starts(starts_arrays, abscodes,abstimes,shift = 4.5):
    #encoding = 'utf-8'
    labels = ['left_paw','right_paw','nose','lever','right_ear','jaw','tongue']

    # convert abspositiosn to integers
    vals = []
    for k in range(abscodes.shape[0]):
        vals.append(np.int(abscodes[k].decode()))
    vals=np.array(vals)

    idx04 = np.where(vals==4)[0]
    idx02 = np.where(vals==2)[0]

    #print ("Fixed shift between lever and DLC ", shift, " sec")
    ax=plt.subplot(111)
    # plot scatter
    for k in range(len(starts_arrays)):
        #if k==3:
            ax.scatter(starts_arrays[k], starts_arrays[k]*0+k, label=labels[k])
    
    # plot lever codes
    ax.scatter(abstimes[idx04]+shift, idx04*0+8, c='darkblue', label='04 codes')
    ax.scatter(abstimes[idx02]+shift, idx02*0+9, c='darkgreen', label='02 codes')

    # metadata
    ax.legend(title="Behaviours initiated by", fontsize=15)
    plt.xlabel("Time (sec)",fontsize=20)
    plt.yticks([])
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.title("DLC traced behaviour initiations by body part ", fontsize=20)
    plt.show()
    

def find_quiet_periods(starts_arrays, length_quiet_period):

    # discretize time in miliseconds
    time_bins = np.zeros((1400*1000),'int32')
    
    # bin all activity starts;
    ax=plt.subplot(2,1,1)
    for k in range(len(starts_arrays)):
        if k != 2:
            time_bins[np.int32(starts_arrays[k]*1000)]+=1
        else:
            print ("Excluding nose movement from assessment of overall body movement")
    plt.plot(np.arange(time_bins.shape[0])/1E3,time_bins)

    # find quiet periods
    idx = np.where(time_bins==0)[0]

    #length_quiet_period = 6 
    starts = []
    inside = []
    inside.append([])
    starts.append(idx[0])
    ctr=0
    for k in range(1,idx.shape[0],1):
        if (idx[k]-idx[k-1])>1:
            starts.append(k)
            inside.append([])
            ctr+=1
        else:
            inside[ctr].append(int(k-1))

    # Plot durations of quiet periods
    ax=plt.subplot(2,1,2)
    durations = []
    for k in range(len(starts)):
        durations.append((inside[k][-1]-inside[k][0])/1000.)

    plt.scatter(np.array(starts)/1E3,durations)
        #starts = np.array(starts); ends=np.array(ends)
    ##print (starts.shape, ends.shape)

    # find periods above suggested lockout
    lockout_len = length_quiet_period*1000 # do 6sec lockouts
    durations = np.array(durations)
    print ("durations")
    starts = np.array(starts)
    idx = np.where(durations>length_quiet_period)[0]
    print (idx.shape)
    plt.scatter(starts[idx]/1E3,durations[idx])
    plt.title(" Quiet periods  ", fontsize=15)

    plt.suptitle("Quiet period length: "+ str(length_quiet_period)+ "sec;    #: "+str(idx.shape[0]), fontsize=20)
    plt.show()

    print ("# of quiet_periods longer than ", length_quiet_period, "   #", starts[idx].shape)
    
    return starts[idx]/1E3
