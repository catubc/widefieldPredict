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

import glob2

from numba import jit
from sklearn.svm import SVC # "Support vector classifier"
import matplotlib.patches as mpatches


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

    #print (" # of trials: ", data.shape[0])

    areas = np.unique(maskwarp) 
    #print (" # of areas: ", areas.shape)
    
    # work in 1D vectors easier to mask
    maskwarp1D = maskwarp.reshape(-1)

    trial_courses = []
    area_ids = []
    for k in range(data.shape[0]):
        #if k%10==0:
        #    print ("computing trial: ", k)
        time_courses_local = []

        # convert to 1D vector to mask faster
        #data1D = np.float32(data[k].reshape(181,-1))
        data1D = np.float32(data[k].reshape(data[k].shape[0],-1))
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
    #print ("# trials, # areas, # times: ", trial_courses.shape)
    #print ("area ids: ", area_ids.shape)

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
                                             time_window,
                                             trial_courses_fixed, 
                                             trial_courses_fixed_ids,
                                             trial_courses_random_fixed, 
                                             trial_courses_random_ids):
    
    # combine good trials with random trials for training sets:
    #time = 0
    #print ("trial_courses_fixed: ", trial_courses_fixed.shape)
    good_trials = trial_courses_fixed[trial_courses_fixed_ids, :,time:time+time_window].reshape(trial_courses_fixed_ids.shape[0], -1)
    print ("good trials: ", good_trials.shape)
    temp = np.arange(trial_courses_fixed.shape[0])
    idx = np.delete(temp,trial_courses_fixed_ids) # remove the training set trials for testing stage
    test_trials = trial_courses_fixed[idx, :,time:time+time_window].reshape(idx.shape[0], -1)  # select left over trials for testing;

    
    random_trials = trial_courses_random_fixed[trial_courses_random_ids, :,time:time+time_window].reshape(trial_courses_random_ids.shape[0], -1)
    print ("random_trials: ", random_trials.shape)
    temp = np.arange(trial_courses_random_fixed.shape[0])
    idx = np.delete(temp,trial_courses_random_ids) # remove training set trials for testing
    test_trials_random = trial_courses_random_fixed[idx, :,time:time+time_window].reshape(idx.shape[0], -1)
    #print ("test_trials_random: ", test_trials_random.shape)

    # make labels
    #y = np.zeros(good_trials.shape[0]+random_trials.shape[0],'int32')
    #y[:good_trials.shape[0]]=1
    y = np.zeros(good_trials.shape[0]+random_trials.shape[0],'int32')
    y[:good_trials.shape[0]]=1

    # concatenate
    X = np.vstack((good_trials,random_trials))

    #rint ("done time: ", time)
    
    
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


def normalize_data(data1, data2, random_flag):
    
    #print ("NIORMALIZEION: ", data1.shape, data2.shape)
    data_in = np.vstack((data1,data2))    
    #print ("Data in: ", data_in.shape)
    
    if random_flag:
        idx_random = np.random.choice(np.arange(data_in.shape[0]), size=data_in.shape[0], replace=False)
        #print (idx_random.shape)
        data_in = data_in[idx_random]
        #print ("Data in: ", data_in.shape)

    
    #data_in shaep: (63, 35, 181)
    for k in range(data_in.shape[1]):
        for p in range(data_in.shape[2]):
            temp = data_in[:,k,p]
            #print ("temp re: ", temp)
            temp = (temp-temp.mean(0))/(temp.std(0)+0.00000001) # to avoid nans
            #print ("temp post: ", temp)
            data_in[:,k,p] = temp
    
    data1 = data_in[:data1.shape[0]]
    data2 = data_in[data1.shape[0]:]   
    #print ("POST NIORMALIZEION: ", data1.shape, data2.shape)

    return data1, data2


def parallel_svm_multiple_tests(time, 
                                time_window,
                                trial_courses_fixed, 
                                trial_courses_fixed_ids,
                                trial_courses_random_fixed, 
                                trial_courses_random_ids,
                                random_flag,
                                root_dir):


   # 
    #print ("normalizing data... (TO CHECK THIS STEP)")
    trial_courses_fixed, trial_courses_random_fixed = \
                normalize_data(trial_courses_fixed, trial_courses_random_fixed, random_flag)
    
    print ("trial_courses_fixed: ", trial_courses_fixed.shape)
    res1 = []
    res2 = []
    sens = []
    spec = []
    accuracy = []
    for k in range(len(trial_courses_fixed_ids)):
        X, y, test_trials, test_trials_random = make_training_sets_multiple_tests_window(time, 
                                                                                         time_window,
                                                                                        trial_courses_fixed, 
                                                                                        trial_courses_fixed_ids[k],
                                                                                        trial_courses_random_fixed, 
                                                                                        trial_courses_random_ids[k])
        #print ("test trials: ", test_trials.shape)
        #print ("X: ", X.shape)
        #print ("y: ", y.shape)
        
        model = SVC(kernel='linear', C=1)
        model.fit(X, y)

        # test real data
        yfit = model.predict(test_trials)
        true_pos = np.sum(yfit)
        false_neg = (test_trials.shape[0]-true_pos)
        res1.append(np.sum(yfit)/float(yfit.shape[0]))

        # test random data
        yfit = model.predict(test_trials_random)
        false_pos = np.sum(yfit)
        true_neg = (test_trials_random.shape[0]-false_pos)
        res2.append(np.sum(yfit)/float(yfit.shape[0]))
    
        # compute sensitivity:  true positives / (true positives + false negatives)
        sens.append(true_pos / (true_pos+false_neg))
        spec.append(true_neg / (true_neg+false_pos))
        
        all_pos = true_pos + false_neg
        all_neg = false_pos + true_neg
        # print ("true_pos :", true_pos)
        # print ("true_neg :", true_neg)
        # print ("false_neg :", false_neg)
        # print ("false_pos :", false_pos)
        # print ("all_pos :", all_pos)
        # print ("all_neg :", all_neg)

        # compute accuracy:
        accuracy.append((true_pos + true_neg)/(all_pos+all_neg))
    
    #return (res1, res2, sens, spec)
    np.save(root_dir + str(time)+'_sens.npy', sens)
    np.save(root_dir + str(time)+'_spec.npy', spec)
    np.save(root_dir + str(time)+'_accuracy.npy', accuracy)
    
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


# def parallel_svm_multi_time_multi_area(times, 
                                                       # areas,
                                                       # trial_courses_fixed, 
                                                       # trial_courses_random_fixed):
    # print ("times ", times)
    
    # res1_array = []
    # res2_array = []
    
    # for area_id in areas:
        # X, y, test_trials, test_trials_random = make_training_sets_multi_times_multi_areas(
                                                                               # times, 
                                                                               # area_id,
                                                                               # trial_courses_fixed, 
                                                                               # trial_courses_random_fixed)

        # model = SVC(kernel='linear', C=2)
        # model.fit(X, y)

        # # test_trials_rewarded = trial_courses_fixed[50:, :,time].reshape(50, -1)
        # # model = grid.best_estimator_
        # yfit = model.predict(test_trials)
        # res1 = np.sum(yfit)/float(yfit.shape[0])
        # res1_array.append(res1)

        # yfit = model.predict(test_trials_random)
        # res2 = np.sum(yfit)/float(yfit.shape[0])
        # res2_array.append(res2)

    # print ("done times: ", times)
    # return (res1_array, res2_array)


# plotting real data;
def plot_true_vs_random(res, trial_courses_fixed, time_window):
        
    sample_rate = 30
    window_size = trial_courses_fixed.shape[2]//sample_rate//2
    
    real_res = np.vstack(res[::2])
    print (real_res)
    random_res = np.vstack(res[1::2])

    fig=plt.figure()
    ax=plt.subplot(111)
    # PLOTS FOR SINGLE AREA / CUMULATIVE AREAS
    t = np.arange(real_res.shape[0])/sample_rate-window_size

    # plot real pulls time-series
    y = real_res.mean(1)
    e = real_res.std(1)
    plt.errorbar(t, y, e, linestyle='None', marker='^', c='blue')
    plt.plot(t, y,c='blue')

    # plot randomized prediction time-series
    y = random_res.mean(1)
    e = random_res.std(1)
    plt.errorbar(t, y, e, linestyle='None', marker='^', c='red')
    plt.plot(t,y,c='red')

    # labels
    plt.title("Decoding rewarded trials (blue); random trials (red)",fontsize=20)
    plt.plot([-window_size,window_size],[0.5,0.5],'r--',c='black')
    plt.plot([0,0],[0,1],'r--',c='black')
    plt.xlabel("Time (sec)",fontsize=20)
    plt.ylabel("Probability (dash=chance)",fontsize=20)

    labels = []
    patch_j = mpatches.Patch(color = 'blue', label = "Rewarded trials")
    labels.append(patch_j)
    patch_j = mpatches.Patch(color = 'red', label = "Random trials")
    labels.append(patch_j)
    #plt.legend(handles=labels)
    ax.legend(handles = labels, fontsize=20)

    plt.suptitle("Sliding time window prediction using "+str(time_window)+" frames = "+str(round(time_window/30.,2))+"sec", fontsize=20)
    plt.show()


# plotting plot_specificity_sensitivity;
def plot_specificity_sensitivity(sens, spec, trial_courses_fixed):
    print ("NOT YET COMPLETED...")
    return
    
    sample_rate = 30
    window_size = trial_courses_fixed.shape[2]//sample_rate//2
    
    real_res = res[::2]
    random_res = res[1::2]

    fig=plt.figure()
    ax=plt.subplot(111)
    # PLOTS FOR SINGLE AREA / CUMULATIVE AREAS
    t = np.arange(real_res.shape[0])/sample_rate-window_size

    # plot real pulls time-series
    y = real_res.mean(1)
    e = real_res.std(1)
    plt.errorbar(t, y, e, linestyle='None', marker='^', c='blue')
    plt.plot(t, y,c='blue')

    # plot randomized prediction time-series
    y = random_res.mean(1)
    e = random_res.std(1)
    plt.errorbar(t, y, e, linestyle='None', marker='^', c='red')
    plt.plot(t,y,c='red')

    # labels
    plt.title("Decoding rewarded trials (blue); random trials (red)",fontsize=20)
    plt.plot([-window_size,window_size],[0.5,0.5],'r--',c='black')
    plt.plot([0,0],[0,1],'r--',c='black')
    plt.xlabel("Time (sec)",fontsize=20)
    plt.ylabel("Probability (dash=chance)",fontsize=20)

    labels = []
    patch_j = mpatches.Patch(color = 'blue', label = "Rewarded trials")
    labels.append(patch_j)
    patch_j = mpatches.Patch(color = 'red', label = "Random trials")
    labels.append(patch_j)
    #plt.legend(handles=labels)
    ax.legend(handles = labels, fontsize=20)

    plt.suptitle("Sliding time window prediction using "+str(time_window)+" frames = "+str(round(time_window/30.,2))+"sec", fontsize=20)
    plt.show()




def plot_accuracy(root_dir, length_rec, fig):
    time_window=1
    #fig=plt.figure()
    ax=plt.subplot(111)
    sample_rate = 30
    window_size = length_rec//sample_rate/2
    spec_array = []
    spec_error = []
    sens_array = []
    sens_error = []
    accuracy_array = []
    accuracy_error = []

    for k in range(length_rec):
        #print (k)
        sens = np.load(root_dir+str(k)+"_sens.npy")
        spec = np.load(root_dir+str(k)+"_spec.npy")
        acc = np.load(root_dir+str(k)+"_accuracy.npy")
        #print (sens, spec)

        # plot real pulls time-series
        sens_array.append(sens.mean(0))
        sens_error.append(sens.std(0))

        # plot randomized prediction time-series
        spec_array.append(spec.mean(0))
        spec_error.append(spec.std(0))

        # plot randomized prediction time-series
        accuracy_array.append(acc.mean(0))
        accuracy_error.append(acc.std(0))

    t = np.arange(len(sens_array))/30.-(length_rec//sample_rate/2)
    # plt.errorbar(t, 
    #              sens_array, sens_error, linestyle='None', marker='^', c='blue', alpha=.7)
    # plt.plot(t, sens_array, c='blue', alpha=.7)

    # plt.errorbar(t, 
    #              spec_array, spec_error, linestyle='None', marker='^', c='red', alpha=.7)
    # plt.plot(t, spec_array, c='red', alpha=.7)


    plt.errorbar(t, 
                 accuracy_array, accuracy_error, linestyle='None', marker='^', c='black', alpha=.7)
    plt.plot(t, accuracy_array, c='black', alpha=.7)



    labels = []
    patch_j = mpatches.Patch(color = 'blue', label = "RS")
    labels.append(patch_j)
    patch_j = mpatches.Patch(color = 'red', label = "Motor")
    labels.append(patch_j)
    patch_j = mpatches.Patch(color = 'black', label = "All Cortex")
    labels.append(patch_j)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    #plt.legend(handles=labels)
    ax.legend(handles = labels, fontsize=20)
    plt.ylim(0,1)
    plt.plot([t[0],t[-1]],[0.5,0.5],'r--',c='black')
    plt.plot([0,0],[0,1],'r--',c='black')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    plt.suptitle("Sliding time window prediction using "+str(time_window)+" frames = "+str(round(time_window/30.,2))+"sec", fontsize=20)
    plt.xlabel("Time (sec)",fontsize=20)
    plt.suptitle(root_dir, fontsize=20)
    #plt.show()
    

def fix_trials(trial_courses, trial_courses_random):
    trial_courses_fixed = trial_courses.reshape(trial_courses.shape[0],-1)
    trial_courses_fixed = np.nan_to_num(trial_courses_fixed, nan=9999, posinf=9999, neginf=9999)
    idx = np.where(trial_courses_fixed==9999)

    trial_courses_fixed[idx]=0

    #print ('trial_courses_fixed: ', trial_courses_fixed.shape)
    trial_courses_random_fixed = trial_courses_random.copy().reshape(trial_courses_random.shape[0],-1)
    #print ('trial_courses_random_fixed: ', trial_courses_random_fixed.shape)
    trial_courses_random_fixed[:,idx[1]] = 0

    #ax=plt.subplot(121)
    #plt.title("Event triggered")
    #plt.imshow(trial_courses_fixed, aspect='auto', vmin=-.15, vmax=.15)
    trial_courses_fixed = trial_courses_fixed.reshape(trial_courses.shape)
    #print ("trial_courses_fixed: ", trial_courses_fixed.shape)

    #ax=plt.subplot(122)
    #plt.title("Random triggered")
    #plt.imshow(trial_courses_random_fixed, aspect='auto', vmin=-.15, vmax=.15)
    trial_courses_random_fixed = trial_courses_random_fixed.reshape(trial_courses_random.shape)

    return trial_courses_fixed, trial_courses_random_fixed
