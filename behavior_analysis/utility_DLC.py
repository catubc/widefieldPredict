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
import csv
from tqdm import trange
import glob, h5py

from Specgram.Specgram import Specgram

import glob2

from numba import jit
from sklearn.svm import SVC # "Support vector classifier"

colors = [
'black','blue','red','green', 'cyan','orange',
'brown','slategrey','darkviolet','darkmagenta',
'lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink']

def plot_DLC_traces_vs_lever(shift, traces, labels, root_dir):

    abstimes, abspositions, abscodes  = load_lever_data(root_dir)
    
    clrs = ['green','blue','black']
    # Plot DLC trces
    fig=plt.figure()
    ctr=0
    for k in [0,1,3]:
    #for k in []:
        print (traces[0].shape)
        times = np.arange(traces[k].shape[0])/15.

        trace_out1 = traces[k][:,0]-np.median(traces[k][:,0])
        trace_out2 = traces[k][:,1]-np.median(traces[k][:,1])
        trace_out = np.sqrt(trace_out1**2 + trace_out2**2)
        if k==3: 
            trace_out*=4.6
        if k==1:
            trace_out*=2.0

        plt.plot(times, trace_out, color=clrs[ctr], linewidth=6, label=labels[k] + " (DLC)", alpha=.7)

        ctr+=1
    # plot lever
    abstimes_shifted = abstimes + shift
    plt.plot(abstimes_shifted, abspositions, 'r--', linewidth=6,  c='red', label="lever (motor)")

    # plot metadata
    plt.ylabel("Lever motoro displacement",fontsize=40)
    plt.xlabel("Time (sec)",fontsize=40)
    plt.ylim(0,100)
    plt.legend(fontsize=30)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)

    plt.show()
    
class Analyze():

    def __init__(self):

        pass

#
def get_ordered_fnames_stand_alone(A): # load ordered sessions from file

    data = np.load(os.path.join(A.root_dir, A.animal_id,'tif_files.npy'))

    #
    sessions = []
    for k in range(len(data)):
        sessions.append(os.path.split(data[k])[1].replace('.tif',''))

    A.sessions = sessions

    return A

def feature_triggered_trace_plots(n_features, trace_id, 
                                  name_feature, traces, 
                                  starts_04, window, shift,
                                  clrs):
    
    triggered_traces = []
    print ("LEN TRACES: ", len(traces))
    for k in range(len(traces)):
        ax=plt.subplot(n_features,7,k+1 +trace_id*7)

        # increase power by 2 for computation of specgram;
        feature3 = traces[k]
        #print ("feature1: ", feature1.shape)  #look at x or y coordinate; x = 0?

        #
        t = np.arange(feature3.shape[0])/15.-shift
        #print ("times; ", t)
        
        x = np.arange(window*30)/15.-window
        # save averages and individaul traces;
        aves = []
        for p in range(starts_04.shape[0]):
            idx = np.where(t>starts_04[p])[0]
            #print (idx)
            if idx.shape[0]==0:
                continue
            temp3 = feature3[idx[0]-window*15:idx[0]+window*15]
            
            if temp3.shape[0]==0:
                continue

            means = temp3
            
            if False:
                plt.plot(x,means, c=clrs[trace_id],alpha=.01)

            if means.shape[0]==x.shape[0]:
                aves.append(np.abs(means))
                           
        #
        medians = np.median(np.vstack(aves),axis=0)
        triggered_traces.append(medians)

        # 
        aves2 = np.vstack(aves)
        aves_mean = aves2.mean(0)
        plt.plot(x,aves_mean, c=clrs[k])

        # PLOT ERROR SHADED
        error = np.std(aves2, axis=0)#/len(aves)
        plt.fill_between(x, aves_mean-error, aves_mean+error, color=clrs[k], alpha=.05)

        # PLOT LABELS
        lims = max(np.max(aves_mean),abs(np.min(aves_mean)))
        plt.ylim(0, np.max(aves_mean+error)*1.1)
        
        if trace_id==0:
            plt.title(labels[k])
            
        if k==0:
            plt.ylabel(name_feature)
        
        if trace_id!=(n_features-1):
            plt.xticks([])
        
        plt.plot([0,0],[-lims*1.5, lims*1.5],'r--',c='black')
        plt.plot([x[0],x[-1]],[0, 0],'r--',c='black')
                    
    
    return triggered_traces
    




def correlate_DLC_vs_motor(root_dir, traces,plotting):
    
    from scipy import stats
    import scipy

    fps = 15.
    
    abstimes, abspositions, abscodes = load_lever_data(root_dir)

    # compute trace of lever for matching;
    trace = np.sqrt(traces[3][:,0]**2+traces[3][:,1]**2)
    trace= np.abs(trace-trace.mean())
    
    # shrink values a bit
    abspositions /=5.

    # UPSAMPLE LEVER MOTOR DATA TO 1MS RESOLUTION
    abspos_upsampled = np.zeros((int(abstimes[-1]*1000)+1),'float32')
    for k in range(0, abstimes.shape[0]-1,1):
        abspos_upsampled[int(abstimes[k]*1000): 
                         int(abstimes[k+1]*1000)]=abspositions[k]
    
    # UPSAMPLE DLC DATA TO 1MS RESOLUTION
    print ("Video frame rate is: ", fps)
    x = np.arange(trace.shape[0])/fps
    traces_upsampled = np.zeros((int(x[-1])*1000+2),'float32')
    for k in range(0,x.shape[0]-1,1):
        traces_upsampled[int(x[k]*1000):int(x[k+1]*1000)]=trace[k]
   
    
    # DETERMINE # OF SHIFTS BETWEEN TRACES FOR FITTING
    cor=[]
    start_ = -10000
    end_ = 10000

    
    # correct DLC very large jumps by setting to median value
    if True:
        p = np.percentile(traces_upsampled, 98) # return 50th percentile, e.g median.
        idx = np.where(traces_upsampled>p)[0]

        # correct values
        median_ = np.median(traces_upsampled)
        traces_upsampled[idx]=median_
    
    temp_norm= np.correlate(traces_upsampled[:abspos_upsampled.shape[0]],
                            traces_upsampled[:abspos_upsampled.shape[0]], mode='valid')
    for k in range(start_,end_,1):
        temp = np.correlate(traces_upsampled[:abspos_upsampled.shape[0]], 
                                      np.roll(abspos_upsampled,k), mode='valid')
        cor.append(temp[0]/temp_norm)          
    
    if plotting:
        ax=plt.subplot(1,2,1)
        plt.plot(cor)

    idx = np.argmax(cor)
    shift = -(idx+start_)/1000.
    
    # plot upsampled DLC traces
    if plotting:
        ax=plt.subplot(1,2,2)
        plt.plot(np.arange(traces_upsampled.shape[0])/1000.+shift, traces_upsampled, c='red')

        # plot absolute positions upsampled
        plt.plot(np.arange(abspos_upsampled.shape[0])/1000., abspos_upsampled, c='black')
        plt.xlim(300, 400)

    # Also do randomized shifts
    cor_random=[]
    shifts_random = np.random.uniform(-20000,20000,size=500)
    #temp_norm= np.correlate(traces_upsampled[:abspos_upsampled.shape[0]],
    #                        traces_upsampled[:abspos_upsampled.shape[0]], mode='valid')
    for k in range(shifts_random.shape[0]):
        temp = np.correlate(traces_upsampled[:abspos_upsampled.shape[0]], 
                                      np.roll(abspos_upsampled,int(shifts_random[k])), mode='valid')

        cor_random.append(temp[0]/temp_norm)    
    

    # ***************************** 
    # COMPUTE RANDOM SHIFT PROBABILITY
    from statsmodels.distributions.empirical_distribution import ECDF

    sample = np.array(cor_random).squeeze()
    #print ("Sample: ", sample.shape)

    # fit a cdf
    ecdf = ECDF(sample)

    # get cumulative probability for values
    idx = np.argmax(cor)
    max_cor = cor[idx]
    print ("Max correlation: ", max_cor)

    
    # shift is negative
    shift = -shift
    
    return cor, shift, cor_random, max_cor, abspos_upsampled



def plot_DLC_scatter(traces_nan):
    import cv2

    fname = '/media/cat/4TBSSD/yuki/IA2/video_files/IA2pm_Apr22_Week2_30Hz.m4v'
    fname = '/media/cat/4TBSSD/yuki/IA1/m4v/prestroke/IA1am_May6_Week4_30Hz.m4v'
    print ("LOADING A HARDFIXED DATASET: ", fname)

    vc = cv2.VideoCapture(fname)
    c=1

    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        #cv2.imwrite(str(c) + '.jpg',frame)
        c = c + 1
        cv2.waitKey(1)

        if c==1000:
            break

    vc.release()

    clrs1 = ['green','blue','brown','black','red','cyan','magenta']

    fig=plt.figure()
    for k in range(7):
        # convert name color to RGB and add float
        temp = np.float32(matplotlib.colors.to_rgba(clrs1[k], alpha=1))
        temp = np.repeat(temp[None,:], traces_nan[k].shape[0],axis=0)
        temp[:,3]= np.arange(0,traces_nan[k].shape[0],1)/float(traces_nan[k].shape[0])/35.

        plt.scatter(traces_nan[k][:,0], traces_nan[k][:,1],c=temp)

    plt.imshow(frame,aspect='auto',alpha=.5)
    plt.ylim(360,0)
    plt.xlim(0,640)

    plt.show()

  

def violion_plots_all_sessions(root_dirs):
    
    import matplotlib.patches as mpatches
    labels = []
    shifts = []

    fig=plt.figure()
    ax=plt.subplot(111)
    plotting=False
    ctr=0
    for root_dir in root_dirs:

        # load traces and labels from DLC .csv file
        fname = glob2.glob(root_dir + '/*1030000.csv')[0]
        traces, labels, traces_nan = load_csv(fname)

        (cor, shift, cor_random, max_cor, abspos_upsampled) = correlate_DLC_vs_motor(root_dir, traces,
                                                                plotting)
        print (" SHIFT FROM CORRELATION: ", shift, " sec")

        cor_random = np.array(cor_random).squeeze()

        vp = plt.violinplot(cor_random, [ctr+0.5], points=20, widths=.75,
                              showextrema=False, showmedians=False)
                              #showmeans=True, , showmedians=True)

        for pc in vp['bodies']:
            pc.set_facecolor('blue')
            pc.set_edgecolor('black')

        plt.scatter(ctr+0.5,max_cor, s=100, c='red')

        ctr+=1
        print('')

    patch = mpatches.Patch(color='blue', label='Random shifts')
    labels.append(patch)
    patch = mpatches.Patch(color='red', label='Best match')
    labels.append(patch)

    #plt.legend(handles=labels, fontsize=30)

    plt.ylim(0,1)
    plt.xticks([0.5,1.5],['apr22','may2'])
    plt.ylabel("Correlation (DLC vs Lever)",fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    plt.show()
    
    return shifts



# fix huge jumps and replace low probability scores with previous location;
def fix_trace(trace, prob_threshold = 0.2):

    probs = trace[:,2]

    idx = np.where(probs<=prob_threshold)[0]
    if idx.shape[0]>0:
        if idx[0]==0:  # skip zeroth timestep in case it has low prob
            idx=idx[1:]

    # replace low probability locations with previous location known
    # OPTIONAL: can reaplce with median over all data;
    for id_ in idx:
        trace[id_,:2]=trace[id_-1,:2]

    return trace

def find_video_to_lever_shift(fname_h5,
                              root_dir,
                              labels,
                              n_processes):

    ############################################################
    ########### COMPUTE SHIFT BETWEEN LEVER AND VIDEO ##########
    ############################################################
    # load traces and labels from DLC .csv file
    try:
        fname_traces=glob.glob(fname_h5.replace('h5',".npy"))[0]
        print (len(fname_traces))
    except:
        h5_to_npy(fname_h5)
        fname_traces=glob.glob(fname_h5.replace('h5',"npy"))[0]
        print (len(fname_traces))


    # FIND SHIFT BY CORRELATING LEVER-MOTOR WITH RIGHT PAW
    traces, traces_nan = load_npy(fname_traces)
    shift = correlate_DLC_vs_motor2(root_dir,
                                    traces,
                                    labels,
                                    n_processes,
                                    plotting=False)
    print (" SHIFT FROM CORRELATION: ", shift, " sec")

    # save shift
    np.savetxt(root_dir+'shift.txt', [shift])

    #
    return shift


def generate_movement_starts(session,
                             root_dir,
                             animal_id,
                             lockout_window,
                             labels,
                             n_processes=10,
                             make_sample_video=False):

    # first, check if session has a video and it has been processed:
    print (animal_id, session)

    session_dir = os.path.join(root_dir,
                               animal_id,
                               'tif_files',
                               session)

    #
    fname_h5 = glob.glob(session_dir+"/*.h5")
    if len(fname_h5)!=1:
        print ("Couldn't find video / processed file, skipping")
        return
    fname_h5 = fname_h5[0]

    #
    fname_out = os.path.join(root_dir, animal_id,'tif_files',
                             session,
                             session+ "_"+str(lockout_window)+"secNoMove_movements.npz")

    if os.path.exists(fname_out)==False:

        ###################################################
        ###################################################
        ###################################################
        # get time shift in data from the
        session_dir = os.path.join(root_dir, animal_id,'tif_files', session)+'/'
        video_shift = find_video_to_lever_shift(fname_h5,
                                                session_dir,
                                                labels,
                                                n_processes)
        #video_shift = -video_shift

        ###################################################
        ###################################################
        ###################################################
        # get movements from the data
        plotting=False
        movements = get_movements(fname_h5,
                                  labels,
                                  plotting)


        # compute movement starts
        fps = 15.
        feature_chunks = []
        for k in range(movements.shape[0]):
            temp = movements[k]
            idx = np.where(temp>0)[0]

            #
            chunks = []
            for p in range(1,idx.shape[0],1):
                if (idx[p]-idx[p-1])>= (lockout_window*fps):
                    #chunks.append([idx[p-1]/fps+video_shift, idx[p]/fps+video_shift])
                    chunks.append([idx[p-1]/fps, idx[p]/fps])

            feature_chunks.append(chunks)
            print (labels[k], "  # of quiescent periods: ", len(chunks))

        # compute quiescent periods across all features
        temp = movements.sum(0)
        idx = np.where(temp>0)[0]
        all_chunks = []
        for p in range(1,idx.shape[0],1):
            if (idx[p]-idx[p-1])>= (lockout_window*fps):
                # all_chunks.append([idx[p-1]/fps+video_shift, idx[p]/fps+video_shift])
                all_chunks.append([idx[p-1]/fps, idx[p]/fps])

        print ("  # of complete quiescent periods: ", len(all_chunks))

        np.savez(fname_out,
                 feature_quiescent = feature_chunks,
                 feature_initiations = movements,
                 all_quiescent = all_chunks,
                 labels = labels,
                 video_shift = video_shift
                 )

        if make_sample_video:
            start = 0
            end = 1000

            #
            idx = fname_h5.find('DLC')
            fname_video = os.path.split(fname_h5[:idx])[1]+".mp4"
            fname_video = os.path.join(root_dir, animal_id,'vids/prestroke/', fname_video)
            print ("FNAME)_VIDEO: ", fname_video)

            #
            traces = np.load(fname_h5.replace('h5','npy'))
            make_video_dlc(traces,
                           movements,
                           fname_video,
                           start,
                           end)

    else:
        print ("data alerady processed")
    #return feature_chunks, all_chunks




def get_movements(fname_vid,
                  labels,
                  plotting=False):

    # load traces and labels from DLC .csv file
    try:
        fname_traces=glob.glob(fname_vid.replace('h5',"npy"))[0]
        print (len(fname_traces))
    except:
        h5_to_npy(fname_vid)
        fname_traces=glob.glob(fname_vid.replace('h5',"npy"))[0]
        print (len(fname_traces))

    traces = np.load(fname_traces)

    # set movement threshold in pixels/frame;
    # NOTE THE DEFAULT 5 WAS TESTED ON 620 x 360 vids; n
    movement_threshold = 5  # number of pixels per frame inidicating movement from stationarity
    max_x = int(np.max(traces[:,:,0]))
    max_y = int(np.max(traces[:,:,1]))
    print ('vid size : ', max_x, max_y)

    # scale threshold for detection
    if max_x > 660:
        #print ("increasing movement threhsold for 1280 x 720 vid")
        movement_threshold = int(movement_threshold*max_x/660)


    #
    # traces_filtered = []
    # starts_arrays_lockout = []  # this tracks only beginning of a movement following a lockout period
    movements = np.zeros((traces.shape[0],traces.shape[1])) # This tracks any change in movement.
    for k in range(traces.shape[0]):

        # load and remove low probability traces
        trace = fix_trace(traces[k])

        # compute velocity and median value
        vel = np.sqrt((trace[1:,0]-trace[:-1,0])**2+
                      (trace[1:,1]-trace[:-1,1])**2)
        idx = np.where(vel<=1)[0]
        vel[idx]=np.nan
        # median_vel = np.nanmedian(vel)
        # print ("velocity: ", vel.shape, "  median vel: ", median_vel)

        #
        #
        idx2 = np.where(vel>=movement_threshold)[0]  # VELOCITY > 3 pixels per time step
        movements[k,idx2]+=1

        # if plotting:
        #     bins = np.arange(0,30,3)
        #     y = np.histogram(vel,bins=bins)
        #     plt.plot(y[1][:-1],y[0], label=labels[k])

    # if plotting:
    #     plt.legend()

    if plotting:
        fig=plt.figure()
        plt.imshow(movements,
                  aspect='auto',
                  interpolation='none',
                  extent=[0,movements.shape[1]/15,0,movements.shape[0]], origin='lower')

        plt.yticks(np.arange(movements.shape[0])+.5,
                  labels)

        plt.show()

    return movements

#
def get_starts_arrays2(window,
                       fname_traces,
                       labels,
                       plotting=False):
    

    # load traces and labels from DLC .csv file
    traces = np.load(fname_traces,allow_pickle=True)

    #
    fps = 15.
    traces_filtered = []
    starts_arrays_lockout = []  # this tracks only beginning of a movement following a lockout period
    starts_arrays_all = [] # This tracks any change in movement.
    for k in range(traces.shape[0]):
        feature_name = k
        #derivative_treshold_local = der_threshold_array[k]

        trace = traces[k]
        trace = fix_trace(trace)

        feature = np.sqrt(trace[:,0]**2+
                          trace[:,1]**2)

        # SKIP FILTER STEP;
        #feature = np.abs(butter_bandpass_filter(feature, 1, 7,15))
        traces_filtered.append(feature)

    #     else:
        starts_local, all_starts = find_starts_lockout_rawtraces_dynamic(feature,
                                                             feature,
                                                              window, 
                                                              labels,
                                                              feature_name,
                                                              plotting)

        print ("feature: ", labels[k], "starts local: ", len(starts_local))
        starts_arrays_lockout.append(starts_local)
        starts_arrays_all.append(all_starts)
        #break
        
    return starts_arrays_lockout, starts_arrays_all, traces_filtered

    
from scipy import signal
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_DLC_traces_vs_lever2(shift, traces, labels, root_dir):

    abstimes, abspositions, abscodes  = load_lever_data(root_dir)
    
    clrs = ['green','blue','black']
    # Plot DLC trces
    fig=plt.figure()
    ctr=0

    # PLOT LEFT PAW, RIGHT PAW and LEVER
    for k in [0,1,6]:
        times = np.arange(traces.shape[0])/15.

        trace_out1 = traces[:,k,0]-np.median(traces[:,k,0])
        trace_out2 = traces[:,k,1]-np.median(traces[:,k,1])
        trace_out = np.sqrt(trace_out1**2 + trace_out2**2)
        if k==3: 
            trace_out*=4.6
        if k==1:
            trace_out*=2.0

        plt.plot(times, trace_out, color=clrs[ctr], linewidth=6, label=labels[k] + " (DLC)", alpha=.7)

        ctr+=1

    # PLOT MOTOR WITH THE SHIFT
    abstimes_shifted = abstimes + shift
    plt.plot(abstimes_shifted, abspositions, 'r--', linewidth=6,  c='red', label="lever (motor)")

    # plot metadata
    plt.ylabel("Lever motoro displacement",fontsize=40)
    plt.xlabel("Time (sec)",fontsize=40)
    plt.ylim(0,100)
    plt.legend(fontsize=30)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)

    plt.show()
    
    


def correlate_DLC_vs_motor2(root_dir, 
                            traces,
                            labels,
                            n_processes,
                            plotting):
    
    fname_out = root_dir+'correlate.npz'
    
    if os.path.exists(fname_out)==False:
        
        from scipy import stats
        import scipy
        from tqdm import trange    

        fps = 15.
        
        abstimes, abspositions, abscodes = load_lever_data(root_dir)

        #correlation_id = 6 # LEVER <- not labeled as well as RIGHT PAW
        correlation_ids = [0,1,6]
        shifts = []
        cors = []

        # shrink values a bit
        abspositions = abspositions/5.

        # UPSAMPLE LEVER MOTOR DATA TO 1MS RESOLUTION
        abspos_upsampled = np.zeros((int(abstimes[-1]*1000)+1),'float32')
        for k in range(0, abstimes.shape[0]-1,1):
            abspos_upsampled[int(abstimes[k]*1000):
                             int(abstimes[k+1]*1000)]=abspositions[k]

        # use lever velocity - THIS DOES NOT WORK WELL
        # if False:
        #     abspos_vel = (abspos_upsampled[1:]-abspos_upsampled[:-1])
        #     abspos_upsampled = abspos_vel

        start_ = -15000
        end_ = 15000
        for correlation_id in correlation_ids:
            #correlation_id = 1 # RIGHT PAW

            # use feature velocity
            # if False:
            #     trace = np.sqrt(traces[:,correlation_id,0]**2+
            #                     traces[:,correlation_id,1]**2)
            #     trace= np.abs(trace-trace.mean())
            # else:
            # use velocities:
            vel = np.sqrt((traces[1:,correlation_id,0]-traces[:-1,correlation_id,0])**2+
                            (traces[1:,correlation_id,1]-traces[:-1,correlation_id,1])**2)
            trace = vel


            # UPSAMPLE DLC DATA TO 1MS RESOLUTION
            #print ("Video frame rate is: ", fps)
            x = np.arange(trace.shape[0])/fps
            traces_upsampled = np.zeros((int(x[-1])*1000+2),'float32')
            for k in range(0,x.shape[0]-1,1):
                traces_upsampled[int(x[k]*1000):int(x[k+1]*1000)]=trace[k]

            # correct DLC very large jumps by setting to median value
            if True:
                p = np.percentile(traces_upsampled, 98) # return 50th percentile, e.g median.
                idx = np.where(traces_upsampled>p)[0]

                # correct values
                median_ = np.median(traces_upsampled)
                traces_upsampled[idx]=median_

            # get normalization factor
            temp_norm= np.correlate(traces_upsampled[:abspos_upsampled.shape[0]],
                                    traces_upsampled[:abspos_upsampled.shape[0]], mode='valid')

            # trim both time series to make sure they are identical otherwise gets complicated
            max_len = min(traces_upsampled.shape[0],
                          abspos_upsampled.shape[0])
            traces_upsampled = traces_upsampled[:max_len]
            abspos_upsampled = abspos_upsampled[:max_len]

            # print (traces_upsampled.shape, abspos_upsampled.shape)
            # COMPUTE CORRELATION; splitting IDS is not faster
            ids = np.arange(start_,end_,1)
            cor = parmap.map(correlate_lever_dlc,
                              ids,
                              temp_norm,
                              traces_upsampled,
                              abspos_upsampled,
                              pm_processes=n_processes,
                              pm_pbar=True)

            cors.append(cor)
            #cors.append(cor)
            if plotting:
                ax=plt.subplot(1,2,1)
                plt.plot(cor)

            idx = np.argmax(cor)
            shift = -(idx+start_)/1000.

            # plot upsampled DLC traces
            if plotting:
                ax=plt.subplot(1,2,2)
                plt.plot(np.arange(traces_upsampled.shape[0])/1000.+shift, traces_upsampled, c='red')

                # plot absolute positions upsampled
                plt.plot(np.arange(abspos_upsampled.shape[0])/1000., abspos_upsampled, c='black')
                plt.xlim(300, 400)

            # Also do randomized shifts and compute average correlation
            cor_random=[]
            shifts_random = np.random.uniform(-20000,20000,size=500)
            for k in trange(shifts_random.shape[0]):
                temp = np.correlate(traces_upsampled,
                                              np.roll(abspos_upsampled,int(shifts_random[k])), mode='valid')

                cor_random.append(temp[0]/temp_norm)

            # shift is negative
            shift = -shift
            shifts.append(shift)


        np.savez(fname_out,
            cors = cors,
            shift = shifts,
            cor_random = cor_random,
            abspos_upsampled = abspos_upsampled
            )
    
    
    else:
        data = np.load(fname_out)
        shift = data['shift']

    return shift



def load_npy(fname):
    '''

    '''

    #
    data_array = np.load(fname)

    # 
    #labels = ['left_paw','right_paw','nose','lever','right_ear','jaw','tongue']
    traces = []
    traces_nan = []
    # zero out low quality DLC values
    threshold = 0.9
    for idx in range(data_array.shape[1]):
        temp = data_array[:,idx]

        # replace low likelihoods with median over the whole stack; as if animal didn't move;
        likelihoods = temp[:,2]
        idx = np.where(likelihoods<threshold)[0]
        temp[idx,0]=np.median(temp[:,0])
        temp[idx,1]=np.median(temp[:,1])
        traces.append(temp.copy())
        
        temp[idx,0]=np.nan
        temp[idx,1]=np.nan
        traces_nan.append(temp.copy())
        
    traces = np.array(traces)
    traces_nan = np.array(traces_nan)
    
    return traces, traces_nan


def h5_to_npy(fname_h5):

    fname_out = fname_h5.replace('.h5',".npy")
    f = h5py.File(fname_h5, "r")

    # grab the relevant data from h5 format
    data = f['df_with_missing']['table']

    # add correct traces
    traces = []
    for k in range(len(data)):
        traces.append(data[k][1].reshape(-1,3))

    traces=np.array(traces).transpose(1,0,2)
    np.save(fname_out, traces)
        
        
        
# find sudden movements in time series
def find_starts_lockout_rawtraces_dynamic(feature, 
                                          feature_filtered,
                                           window, 
                                           labels,
                                           feature_name,
                                           plotting=False):
    
    
    
    fps = 15.
    times = np.arange(feature.shape[0])/fps #- 0.5
    
    # LOOK AT GRADIENT
    der = np.abs(np.gradient(feature))
    der_filtered = np.abs(butter_bandpass_filter(der, 1, 7,15))

    # compute the mean centred traces
    feature_local = np.abs(feature-feature.mean(0))
    
    if plotting:
        fig=plt.figure()
        #plt.plot(times, feature_local, c= 'black')
        plt.plot(times, der, c='red')
        plt.plot(times, der_filtered, 'blue')
                 
        
    # HEURISTIC TO REMOVE WILD JUMPS IN THE DLC TRACKING;
    trace_threshold = 300
    idx_trace = np.where(feature_local<trace_threshold)[0]
    
    # COMPUTE THRESHOLD BASED ON THE FILTETERED FEATURE TO AVOID DLC JUMPS/ERRORS ENTERING INTO STD comptuation
    der_threshold = np.std(der_filtered)/3.
    idx_der = np.where(der>der_threshold)[0] # BUT APPLY IT TO THE UNFILTERED TRACE; OTHERWISE FILTER causes shifts in time.
    all_starts = idx_der/fps

    # COMPUTE OVERLAP BETWEEN DERIVATIVE THRESHOLD CROSSINGS AND LOCATIONS THAT ARE NO DLC ERRORS
    final_idx = np.intersect1d(idx_trace, idx_der)
    if plotting:
        plt.scatter(times[final_idx], final_idx*0+10, c='blue')

    # LOOP OVER EACH LOCATION AND IDENTIFY WHETHER THE TIME FOLLOWS A LONG PERIOD OF LOW/NO-MOVEMENT
    times_out = times[final_idx]
    starts_local = []
    if times_out.shape[0]>0:
        starts_local.append(times_out[0])
        for p in range(1, times_out.shape[0]):
            
            # check if previous movement was > window away
            if (times_out[p]-times_out[p-1])>window:
                
                # If yes, then check no movement in previous X sec; if not, it's a new start
                if np.max(der[int((times_out[p]-window)*fps):int(times_out[p]*fps)])<der_threshold:
                    starts_local.append(times_out[p])

    starts_local = np.array(starts_local)

    # 
    if plotting:
        plt.scatter(starts_local, starts_local*0+11, s=40, c='green')
        plt.title(str(feature_name) + " " + labels[feature_name])
        #plt.xlim(90, 110)
    
        
    return starts_local, all_starts
    
    
def starts_grooming(traces,
                    window,
                   plotting=False):
    
    fps=15.
    threshold_grooming = 200 # Number of pixels of proximity between rigth paw and nose for grooming behaviour

    # find mean location of nose;
    mean_nose_x = np.nanmedian(traces[:,2,0])
    mean_nose_y = np.nanmedian(traces[:,2,1])

    dist_right_paw_to_nose = np.sqrt((traces[:,1,0]-mean_nose_x)**2 + 
                             (traces[:,1,1]-mean_nose_y)**2)
    if plotting:
        y = np.histogram(dist_right_paw_to_nose, bins=np.arange(0,500,25))
        fig=plt.figure()
        plt.title("Distance between nose (mean location) and right paw") 
        plt.plot(y[1][:-1],y[0])
        plt.show()

    starts_grooming = np.where(dist_right_paw_to_nose<200)[0]/fps

    # compute lockouts 
    starts_local = []
    starts_local.append(starts_grooming[0])
    for k in range(1,starts_grooming.shape[0],1):
        if starts_grooming[k]-starts_grooming[k-1]>=window:
            starts_local.append(starts_grooming[k])

    starts_grooming = np.array(starts_local)
    print (" # of groomings: ", starts_grooming.shape)
    
    
    return starts_grooming

from scipy import signal


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


# find sudden movements in time series
def find_starts_lockout_rawtraces(feature,
                                  window,
                                  der_threshold,
                                  plotting=False):
    
    fps = 15.
    times = np.arange(feature.shape[0])/fps #- 0.5
    
    # USE SPECGRAM TO FIND CHANGES
    der = np.abs(np.gradient(feature))

    feature_local = np.abs(feature-feature.mean(0))
    
    if plotting:
        fig=plt.figure()
        plt.plot(times, feature_local, c= 'black')
        plt.plot(times, der, c='red')

    trace_threshold = 5000
    idx_trace = np.where(feature_local<trace_threshold)[0]
    der_threshold = 2
    #print ("mean: ", np.mean(feature_local), " median: ", np.median(feature_local))
    idx_der = np.where(der>der_threshold)[0]
    
    final_idx = np.intersect1d(idx_trace, idx_der)
    #print (final_idx)
    
    if plotting:
        plt.scatter(times[final_idx], final_idx*0+10, c='blue')
    
    times_out = times[final_idx]
    
    starts_local = []
    if times_out.shape[0]>0:
        starts_local.append(times_out[0])
        activity_threshold = 500
        for p in range(1, times_out.shape[0]):

            # check if previous movement was > window away
            if (times_out[p]-times_out[p-1])>window:
                # check if previous 3 sec had any movement
                #print (int(times_out[p]-window)*15, int(times_out[p])*15)
                if np.max(feature_local[int((times_out[p]-window)*15):int(times_out[p]*15)])<activity_threshold:
                    starts_local.append(times_out[p])

    starts_local = np.array(starts_local)

    # 
    if plotting:
        plt.scatter(starts_local, starts_local*0+11, s=40, c='green')
        plt.xlim(90, 110)
    
        
    return starts_local



def get_starts_arrays(root_dir,
                      window,
                      plotting=False):

    
    starts_arrays_lockout = []
    starts_arrays = []

    # load traces and labels from DLC .csv file
    fname = glob2.glob(root_dir + '/*.csv')[0]
    traces, labels, traces_nan = load_csv(fname)

    # compute data 
    x = np.arange(traces[0].shape[0])/15.
    traces_filtered = []
    #window=12.0    # left paw, right paw, nose, lever...
    der_threshold_array = [5,5,1,3,1,1,1]
    for k in range(len(traces)):
    #for k in range(4):
    #for k in [3]:
        #feature = (traces[k][:,0]-np.median(traces[k][:,0]))*2
        feature = np.sqrt(traces[k][:,0]**2+traces[k][:,1]**2)

        #feature = butter_highpass_filter(feature, 1, 15)*3
        traces_filtered.append(feature)

    #     else:
        starts_local = find_starts_lockout_rawtraces(feature*3,
                                                     window,
                                                     der_threshold_array[k],
                                                     plotting)

        print (k, "starts local: ", len(starts_local))
        starts_arrays_lockout.append(starts_local)

        starts_arrays.append(starts_local)
        #break
        
    return starts_arrays, traces_filtered


def make_video_dlc(traces,
                   movements,
                   fname_video,
                   start=0,
                   end=1000):

    ####################################################################
    ################## MAKE LABELED VIDEOS #############################
    ####################################################################
    import cv2
    from tqdm import trange

    #
    clrs = ['blue','red','yellow','green', 'magenta','pink','cyan']

    #
    #traces = np.load(fname_traces)
    print ("Traces: ", traces.shape)
    print ("loadin gmovie: ", fname_video)

    #
    #fname_in = '/media/cat/4TBSSD/yuki_lever-ariadna-2020-07-21/IA1/videos_to_label/prestroke/IA1pm_Feb5_30Hz.mp4'
    original_vid = cv2.VideoCapture(fname_video)

    #
    original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = original_vid.read()
    original_vid.set(cv2.CAP_PROP_POS_FRAMES, start)

    # set video sizes
    try:
        size_vid = np.array([frame.shape[1],frame.shape[0]])
    except:
        print ("Missing video")
        return
    dot_size = 16

    #
    fname_out = fname_video.replace('.mp4','_labeled_'+str(start)+"_"+str(end)+'.mp4')
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    video_out = cv2.VideoWriter(fname_out,fourcc, 25, (size_vid[0],size_vid[1]), True)

    # setup cutoff
    pcutoff = 0.5
    font = cv2.FONT_HERSHEY_PLAIN

    #
    for n in trange(start, end, 1):
        ret, frame = original_vid.read()
        if n==start:
            print ("Frame size read: ", frame.shape)

        cv2.putText(frame, str(n)+ "  " +str(round(n/15.,1)),
                    (50, 50),
                    font, 3,
                    (255, 255, 0),
                    5)

        # loop over the featuers
        for f in range(traces.shape[0]):
            if traces[f,n,2]<pcutoff:
                continue

            # check if movement occured
            if movements[f][n]==1:
                dot_size=15
            else:
                dot_size=5

            #
            x = int(traces[f,n,0])
            y = int(traces[f,n,1])
            frame[y-dot_size:y+dot_size,x-dot_size:x+dot_size]= (np.float32(
                matplotlib.colors.to_rgb(clrs[f]))*255.).astype('uint8')
                #matplotlib.colors.to_rgb('white'))*255.).astype('uint8')

        video_out.write(frame)

    video_out.release()
    original_vid.release()
    cv2.destroyAllWindows()

#
def get_starts_arrays_fname(fname,
                            window,
                            plotting=False):

    #
    starts_arrays_lockout = []
    starts_arrays = []

    # load traces and labels from DLC .csv file
    traces = np.load(fname)

    print ("TODO: filter out low probabilityes and high jumps / replace with mean or NEAREST")

    # compute data
    x = np.arange(traces.shape[1])/15.
    traces_filtered = []

    #window=12.0    # left paw, right paw, nose, lever...
    #der_threshold_array = [5,5,1,3,1,1,1]
    der_threshold_array = [5,5,1,3,1,1,1]
    for k in range(traces.shape[0]):
        #feature = (traces[k][:,0]-np.median(traces[k][:,0]))*2
        feature = np.sqrt(traces[k][:,0]**2+traces[k][:,1]**2)

        #feature = butter_highpass_filter(feature, 1, 15)*3
        traces_filtered.append(feature)

    #     else:
        starts_local = find_starts_lockout_rawtraces(feature,
                                                     window,
                                                     der_threshold_array[k],
                                                     plotting)

        print (k, "starts local: ", len(starts_local))
        starts_arrays_lockout.append(starts_local)

        starts_arrays.append(starts_local)
        #break

    return starts_arrays, traces_filtered




def feature_triggered_trace_plots_baseline(n_features, trace_id, 
                                  name_feature, traces, 
                                  starts_04, window, shift,
                                  clrs):
    
    labels = ['left_paw','right_paw','nose','lever','right_ear','jaw','tongue']

    triggered_traces = []
    test_array = []
    frame_rate = 15.
    
    # PLOT ALL TRACES TRIGGERED ON PARTICULAR FEATURE STARTS PROVIDED IN starts_04
    # loop over all traces
    for k in range(len(traces)):
        ax=plt.subplot(n_features,7,k+1 +trace_id*7)

        # select a feature traces
        feature3 = traces[k]

        #
        t = np.arange(feature3.shape[0])/15.-shift
        
        x = np.arange(window*30)/15.-window
        # save averages and individaul traces;
        all_traces = []
        if name_feature=='code_04':
            test_array.append([])

        for p in range(starts_04.shape[0]):
            idx = np.where(t>starts_04[p])[0]
            #print (idx)
            if idx.shape[0]==0:
                continue

            temp3 = feature3[int(idx[0]-window*15):
                                 int(idx[0]+window*15)]
            
            if temp3.shape[0]==0:
                continue

            # SHOW CHANGE FROM BASELINE 
            trace_dff = temp3 - np.mean(temp3[:window*15])

            if name_feature=='code_04':
                test_array[k].append(temp3)
            
            if trace_dff.shape[0]==x.shape[0]:
                all_traces.append(np.abs(trace_dff))
                           
        #
        medians = np.median(np.vstack(all_traces),axis=0)
        triggered_traces.append(medians)

        # 
        aves2 = np.vstack(all_traces)
        aves_mean = aves2.mean(0)
        plt.plot(x, aves_mean, c=clrs[k])

        # PLOT ERROR SHADED
        error = np.std(aves2, axis=0)#/len(aves)
        plt.fill_between(x, aves_mean-error, aves_mean+error, color=clrs[k], alpha=.05)

        # PLOT LABELS
        lims = max(np.max(aves_mean),abs(np.min(aves_mean)))
        plt.ylim(0, np.max(aves_mean+error)*1.1)
        
        if trace_id==0:
            plt.title(labels[k])
            
        if k==0:
            plt.ylabel(name_feature,fontsize=8)
        
        if trace_id!=(n_features-1):
            plt.xticks([])
        
        plt.plot([0,0],[-lims*1.5, lims*1.5],'--',c='black')
        plt.plot([x[0],x[-1]],[0, 0],'--',c='black')
    
    test_array = np.array(test_array)

    return triggered_traces, test_array


def feature_triggered_trace_plots_baseline_single_panel(
								  trace_id,   					# body # that we are triggering off
                                  traces,    					# all the traces
                                  starts_04,  					# the movement starts for each body feature
                                  included_traces,				# body #s that we are comparing with trace_id
                                  window, 					    # window in seconds for inclusion
                                  shift,						# 
                                  clrs,
                                  labels):
    
    # 
    triggered_traces = []

    # PLOT ALL TRACES TRIGGERED ON PARTICULAR FEATURE STARTS PROVIDED IN starts_04
    # loop over all traces
    fig=plt.figure()
    ax=plt.subplot(1,1,1)
    for k in range(len(traces)):

        if k in included_traces:
            pass
        else:
            continue
        
        # select a feature traces
        feature3 = traces[k]

        #
        t = np.arange(feature3.shape[0])/15.-shift  # VIDEO TIMESCALE
        
        x = np.arange(2*window*15)/15.-window         # 
        
        # save averages and individaul traces;
        aves = []
        for p in range(starts_04.shape[0]):
            idx = np.where(t>=starts_04[p])[0] # look for the nearest start video frame closest to the start

            if idx.shape[0]==0:
                continue

			# trigger of the time but convert back to video timescale wihch is 15FPS
            temp3 = feature3[int(idx[0]-window*15): int(idx[0]+window*15)]
            
            if temp3.shape[0]==0:
                continue

            # SHOW CHANGE FROM BASELINE
            # subtract the previous window average
            # SIMILAR TO DFF/F
            baseline = np.mean(temp3[temp3.shape[0]//2-3*15:temp3.shape[0]//2])
            means = temp3 - baseline
            #means = temp3 - np.min(temp3[:window*15])
            
            if means.shape[0]==x.shape[0]:
                aves.append(np.abs(means))
                           
        #
        medians = np.median(np.vstack(aves),axis=0)
        triggered_traces.append(medians)

        # 
        aves2 = np.vstack(aves)
        aves_mean = aves2.mean(0)
        aves_mean_norm = aves_mean/aves_mean.max(0)  #NORMALIZE
        if k != trace_id:
            plt.plot(x,aves_mean_norm, linewidth=6, c=clrs[k], label=labels[k])

        # PLOT ERROR SHADED
        error = np.std(aves2, axis=0)#/len(aves)
        #plt.fill_between(x, aves_mean-error, aves_mean+error, color=clrs[k], alpha=.05)

        # PLOT LABELS
        #lims = max(np.max(aves_mean),abs(np.min(aves_mean)))
        lims = np.array([0,1.1])
        plt.ylim(lims[0], lims[1])
        
        plt.plot([0,0],[-lims*1.5, lims*1.5],'--',c='black')
        plt.plot([x[0],x[-1]],[0, 0],'--',c='black')
                    
    
    plt.legend(fontsize=30, loc=1)
    plt.xlim(x[0],x[-1])
    plt.ylim(0,1)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.title(labels[trace_id],fontsize=20)

    return triggered_traces
    
#   
def find_first(array1, window=3.0):
    
    idxs = []
    idxs.append(0)
    for k in range(1,array1.shape[0],1):
        if (array1[k]-array1[k-1])>=window:
            idxs.append(k)
    
    idxs=np.array(idxs)
    return idxs
    

# find leading starts
def find_nearest2(val, array2, window=3.0):
    
    diff_temp = np.min(np.abs(array2-val))
    if diff_temp <= window:
        idx = np.argmin(np.abs(array2-val))
        loc = array2[idx]
        diff = array2[idx]-val
    else:
        loc = np.nan
        diff = 1E10
    
    return diff, loc
    
    

def replace_DLC_lever(traces_in, abspos_upsampled, shift):
    # MANUALLY DOWNSAMPLE THE 1MS resolution positions to match video at 15Hz
    temp = []
    for k in range(int(traces_in[3].shape[0])):
        try:
            temp.append(abspos_upsampled[int(k*1000/15)])
        except:
            break
    
    traces_in[3]= np.roll(temp, int(shift*15)) #np.roll(abspos_upsampled.copy()[::1000/15.], int(3.414*15)-25)

    return traces_in


def find_leads(paw_initiations, 
			lever_initiations, 
			window=3.0):
				
    paw_leads = []
    lever_leads = []
    dist1 = []
    dist2 = []
    for k in range(paw_initiations.shape[0]):
        diff, loc = find_nearest2(paw_initiations[k],lever_initiations, window)
        #print (paw_initiations[k], diff, loc)
        
        if (diff >= 0) and (diff <= window):
            paw_leads.append(paw_initiations[k])
            dist1.append(diff)
        elif (diff <= 0) and (diff > -window):
            lever_leads.append(paw_initiations[k])
            dist2.append(diff)

    paw_leads = np.array(paw_leads)
    lever_leads = np.array(lever_leads)

    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    
    return paw_leads, lever_leads, dist1, dist2


     
def plot_number_04_all_mice_all_sessions(main_dir, 
										tag, 
										root_dirs,
										fig=None):
    
    #root_dirs = os.listdir(main_dir)
    
    if fig is None:
        fig=plt.figure()
    
    Yfit = []
    Ydata = []
    X_array = []
    ctr=0
    for root_dir in root_dirs:
        ax=plt.subplot(2,3,ctr+1)
        # run this function to compute the # of trials 
        compute_single_mouse_all_sessions_number_of_04_02_trials(main_dir+'/'+root_dir)

		# 
        data = np.load(main_dir+'/'+root_dir+'/tif_files/no_trials.npz')
        abs_dates = data['abs_dates']
        n_04=data['n_04']
        n_02=data['n_02']
        n_04_smooth=data['n_04_smooth']
        n_02_smooth=data['n_02_smooth']

        # plot individual mice
        x = abs_dates
        if tag=='04':
            y = n_04
        elif tag=='02':
            y = n_02
        else:
            print (" CODE IS INCRRROECT")
            break
           
        if tag=='04':
            plt.scatter(x, y, s=200, 
            c='blue',
             edgecolors='black',
             alpha=.2)
             
        elif tag =='02':
            plt.scatter(x, y, s=200, 
            c='red',
            marker='x',
            edgecolors='black',
            alpha=.4)

        coef = np.polyfit(x,y,1)
        Ydata.append(y)
        poly1d_fn = np.poly1d(coef) 
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        Y = poly1d_fn(x)
        X_array.append(x.copy())
        Yfit.append(Y.copy())
        if tag=='04':
            plt.plot(x, Y, linewidth=5, c='blue',
                 label=" rewarded "+str(root_dir))
        elif tag=='02':
              plt.plot(x, Y, '--', linewidth=5, c='red',
              label=" unrewarded "+str(root_dir))
            

        # print metadata
        #stroke = np.loadtxt(main_dir+'/'+root_dir+'/stroke.txt', dtype='str')
        #plt.title(root_dir+" "+str(stroke),fontsize=10)
        #ctr_animal+=1

        #if ctr == 0:
        #plt.xlabel("Day of study",fontsize=30)
        #plt.ylabel("# of pulls",fontsize=30)
        plt.ylim(bottom=0, top = 250)
        plt.xlim(0,x[-1])
        #else:
        #    plt.xticks([])
        plt.legend(fontsize=20)

        ctr+=1
    
    #ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

    # plt.xlim(0,140)
    # if tag=='02':
        # plt.ylim(0,400)
        # plt.suptitle("Un-Rewarded pulls over time", fontsize=40)
    # else:
        # plt.ylim(0,200)
        # plt.suptitle("Rewarded pulls over time", fontsize=40)

    # plt.show()
    
    return Yfit, Ydata, X_array

def load_lever_data(root_dir):
    temp = root_dir + '*abstimes.npy'
    #print ("TRYING TO LOAD ABSTIMES: ", temp)
    try:
        fname = glob2.glob(temp)[0]
    except:
        return [], [], []

    abstimes = np.load(fname)
    fname = glob2.glob(root_dir + '*abspositions.npy')[0]
    abspositions = np.load(fname)
    fname = glob2.glob(root_dir + '*abscodes.npy')[0]
    abscodes = np.load(fname)

    return abstimes, abspositions, abscodes
    


def plot_lever_traces_02_04_single_animal(root_dir):
    
    abstimes, abspositions, abscodes = load_lever_data(root_dir)

    # flip lever angle for certain datasets
    if np.mean(abspositions)<0:
        abspositions = -abspositions
    
    lockout=3.0
    (starts_04, starts_04_idx,starts_02, starts_02_idx)  = find_code04_starts2(abscodes, abstimes, abspositions, lockout)
    #print (starts_04)

    # fig=plt.figure()
    # t = np.arange(abspositions.shape[0])/120.
    # plt.plot(abstimes[:5000], abspositions[:5000])

    fontsize=30
    start_window = -3
    end_window = 3
    linewidth = 5
    t=abstimes[:(end_window-start_window)*120]+start_window+0.0250
    fig=plt.figure()
    ax= plt.subplot(111)
    total_traces = 0
    for k in range(0,starts_04_idx.shape[0],1):
        try:
            temp = abspositions[starts_04_idx[k]+start_window*120:starts_04_idx[k]+end_window*120]
            if temp[:3*120].max(0)<=40:
                plt.plot(t, temp, linewidth=linewidth, alpha=.5)
                total_traces+=1
            else:
                print ("skipped ",k)
        except:
            print ("Excluding trace")
            pass


    plt.plot([t[0], t[-1]], [0,0],c='black')
    plt.plot([t[0], t[-1]], [11,11], c='black')
    plt.plot([t[0], t[-1]], [40,40],'r--',c='black')
    #plt.plot([t[0], t[-1]], [80,80],'r--',c='black')
    #plt.plot([t[0], t[-1]], [60,60],c='black', label='# rewarded pulls: '+str(starts_04_idx.shape[0]))
    plt.plot([t[0], t[-1]], [60,60],c='black', label='# rewarded pulls: '+str(total_traces) + \
             " of total 04: "+ str(starts_04_idx.shape[0]))
    plt.plot([0, 0], [0,100],'r--', linewidth=3, c='black')
    plt.ylim(0,100)
    plt.xlim(t[0],t[-1])
    plt.ylabel("Lever Angle (AU)",fontsize=fontsize)
    plt.xlabel("Time (sec)",fontsize=fontsize)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)
    plt.legend(fontsize=fontsize,ncol=2, loc=2)
    #plt.title("No of trials "+str())

    # controls
    fig=plt.figure()
    ax=plt.subplot(111)
    for k in range(0,starts_02_idx.shape[0],1):
        if k==0:
            continue
        plt.plot(t, abspositions[starts_02_idx[k]+start_window*120:starts_02_idx[k]+end_window*120], 
                 #linewidth=linewidth, c=colors[k],label="Unrewarded: Maxed Out")
                 linewidth=linewidth, alpha=.5)
    plt.plot([t[0], t[-1]], [0,0],c='black')
    plt.plot([t[0], t[-1]], [11,11], c='black')
    plt.plot([t[0], t[-1]], [40,40],'r--',c='black')
    #plt.plot([t[0], t[-1]], [80,80],'r--',c='black')
    plt.plot([t[0], t[-1]], [60,60],c='black', label="# unrewarded pulls: "+str(starts_02_idx.shape[0]))
    plt.plot([0, 0], [0,80],c='black')
    plt.ylim(0,100)
    plt.xlim(t[0],t[-1])
    plt.ylabel("Lever Angle (AU)",fontsize=fontsize)
    plt.xlabel("Time (sec)",fontsize=fontsize)
    plt.legend(fontsize=fontsize,ncol=2, loc=2)
    ax.tick_params(axis = 'both', which = 'major', labelsize = fontsize)

    plt.show()
    
    
# plot scatter of # of 04 and 02 performances over time

def compute_single_mouse_all_sessions_number_of_04_02_trials(root_dir):
    #root_dirs = os.listdir(main_dir)
    #ctr_animal=0
    #fig=plt.figure()

   # for root_dir in root_dirs:

    tif_root = root_dir + '/tif_files/'
    fname_out = tif_root + "/no_trials.npz"
    
    #print (fname_out)
    if os.path.exists(fname_out)==False:
        tif_files= os.listdir(tif_root)

        # make sure this order makes sense to not loop over wrong way...
        month_names = ['July','Aug','Sep','Oct','Nov','Dec','Jan','Feb', 'Mar','Apr','May','June']

        abs_dates=[]
        n_04=[]
        n_02=[]
        for tif_file in tif_files:
            text = tif_root+ '/'+tif_file+'/*abstimes.npy'

            ctr=0
            month = None
            while True:
                # for name in month_names
                if month_names[ctr] in text:
                    month = ctr
                    break
                ctr+=1
            idx = tif_file.find(month_names[ctr])
            day = tif_file[idx+3:idx+5]
            day = day.replace("_",'')
            day = day.replace('a','')
            day = day.replace('p','')
            day = day.replace('m','')

            day = int(day)

            try:
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abstimes.npy')[0]
                abstimes = np.load(fname)
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abspositions.npy')[0]
                abspositions = np.load(fname)
                fname = glob2.glob(tif_root+ '/'+tif_file+'/*abscodes.npy')[0]
                abscodes = np.load(fname)
            except:

                print ("data missing, skipping...")

                continue
                
                # find where 04 - rewarded pulls start;
            (starts_04, starts_04_idx,starts_02, starts_02_idx) = find_code04_starts2(abscodes, abstimes, abspositions)

            abs_dates.append(month*31+day)
            n_04.append(starts_04.shape[0])
            n_02.append(starts_02.shape[0])

        abs_dates=np.array(abs_dates)
        n_04=np.array(n_04)
        n_02=np.array(n_02)

        idx = np.argsort(abs_dates)
        abs_dates = abs_dates[idx]
        abs_dates=abs_dates-abs_dates[0]
        n_04=n_04[idx]
        n_02=n_02[idx]

        #ax=plt.subplot(3,4,ctr_animal+1)

        # plot scatter
        #plt.scatter(abs_dates, n_04, c='black',alpha=.5)
        #plt.scatter(abs_dates, n_02, c='red',alpha=.5)

        win1 = 10
        n_04_smooth = np.convolve(n_04, np.ones(win1)/win1,mode='same')
        n_02_smooth = np.convolve(n_02, np.ones(win1)/win1,mode='same')

        
        np.savez(fname_out,
                abs_dates = abs_dates,
                n_04=n_04,
                n_02=n_02,
                n_04_smooth=n_04_smooth,
                n_02_smooth=n_02_smooth)

def find_code04_starts2(abscodes, abstimes, abspositions, lockout=3.0):
    vals = []
    
    if type(abscodes[0])==np.int64:
        vals = abscodes
    else:
        for k in range(abscodes.shape[0]):
            vals.append(np.int(abscodes[k].decode()))
        vals=np.array(vals)
        
    #    
    idx04 = np.where(np.logical_and(vals==4, np.abs(abspositions)>40))[0]
    idx02 = np.where(np.logical_and(vals==2, np.abs(abspositions)>40))[0]
    
    # figure out the first place where 04 code is registered;
    starts_04 = []
    starts_04_idx = []
    for k in range(1, abstimes[idx04].shape[0], 1):
        if (abstimes[idx04][k]-abstimes[idx04][k-1])>lockout:
            starts_04.append(abstimes[idx04[k]])
            starts_04_idx.append(idx04[k])
            
    starts_04 = np.array(starts_04)
    starts_04_idx = np.array(starts_04_idx)

    # starts 02 bad performance
    starts_02 = []
    starts_02_idx = []
    for k in range(1, abstimes[idx02].shape[0], 1):
        if (abstimes[idx02][k]-abstimes[idx02][k-1])>lockout:
            starts_02.append(abstimes[idx02][k])
            starts_02_idx.append(idx02[k])

    starts_02 = np.array(starts_02)
    starts_02_idx = np.array(starts_02_idx)
    
    return starts_04, starts_04_idx, starts_02, starts_02_idx


def load_csv(fname):
    with open(fname, newline='') as csvfile:
        data = list(csv.reader(csvfile))

    labels = data[1]
    print ("data labels: ", labels)
    print ("column vals: ", data[2])

    # load values
    data_array = np.array(data[3:])
    print (data_array.shape)

    # 
    labels = ['left_paw','right_paw','nose','lever','right_ear','jaw','tongue']
    traces = []
    traces_nan = []
    # zero out low quality DLC values
    for idx in range(1,20,3):
        temp = np.float32(data_array[:,idx:idx+3])

        # replace low likelihoods with median over the whole stack; as if animal didn't move;
        likelihoods = temp[:,2]
        idx = np.where(likelihoods<0.9)[0]
        temp[idx,0]=np.median(temp[:,0])
        temp[idx,1]=np.median(temp[:,1])
        traces.append(temp.copy())
        
        temp[idx,0]=np.nan
        temp[idx,1]=np.nan
        traces_nan.append(temp.copy())
        

    return traces, labels, traces_nan
        

        

# # find sudden movements in time series
# def find_starts(feature):
    # plotting = True
    

    # times = np.arange(feature.shape[0])/15. #- 0.5
    # #print (times)
    # fs=15
    # P, extent = Specgram(data = feature, sampfreq = fs,
                         # p0=-60, f0=0.5, f1=fs,
                         # width=0.25, tres = 0.125)


    # data1 = P[0]-np.median(P[0])
    # data2 = np.abs(feature)

    # # only need to find this shift once!
    # rolling_data = []
    # for k in range(-20,20,1):
        # #rolling_data.append((data1*np.roll(data2[::2][:-1],k)).sum())
        # rolling_data.append((data1*np.roll(data2[::2][:-1][:data1.shape[0]],k)).sum())

    # #shift = np.argmax(rolling_data)-20
    # shift = 0 #-1
    # #print ("HARDCODED feature shift: ", shift)
    
    # #plot maximum for the roll shift;
    # if plotting:
        # fig = plt.figure()
        # ax = plt.subplot(2,1,1)
        # plt.plot(rolling_data)

    # P_med = P[0]-np.median(P[0])
    # starts = []
    # for k in range(1,P_med.shape[0],1):
        # if (P_med[k]>P_med[k-1]) and (P_med[k-1]==0):
            # starts.append(k)
    # starts = np.array(starts)/15.*2

    
    # # plot feature traces, power-spectrogram peaks, argrelmax peaks and initiation peaks
    # if plotting:
        # locs = np.array(scipy.signal.argrelmax(P_med)).squeeze()/15.*2
        # ax=plt.subplot(2,1,2)
        # plt.plot(times+shift/15., np.abs(feature))
        # plt.plot(times[::2][:-1],P_med, c='red')
        # plt.scatter(locs, P_med[np.int32(locs*15/2.)], c='green')
        # plt.scatter(starts, P_med[np.int32(starts*15/2)]*1.1, c='orange')
        # plt.show()
        # print ("FINISHED PLOTTING") 
    # return starts

  


# find sudden movements in time series
def find_starts_lockout(feature, plotting=False):
    #plotting = True    

    times = np.arange(feature.shape[0])/15. #- 0.5
    
    # USE SPECGRAM TO FIND CHANGES
    fs=15
    P, extent = Specgram(data = feature, sampfreq = fs,
                         p0=-60, f0=0.5, f1=fs,
                         width=0.25, tres = 0.125)


    data1 = P[0]-np.median(P[0])
    data2 = np.abs(feature)

    # only need to find this shift once!
    rolling_data = []
    for k in range(-20,20,1):
        #rolling_data.append((data1*np.roll(data2[::2][:-1],k)).sum())
        rolling_data.append((data1*np.roll(data2[::2][:-1][:data1.shape[0]],k)).sum())

    #shift = np.argmax(rolling_data)-20
    shift = -1
    #print ("HARDCODED feature shift: ", shift)
    
    # #plot maximum for the roll shift;
    # if plotting:
        # fig = plt.figure()
        # ax = plt.subplot(2,1,1)
        # plt.plot(rolling_data)

    P_med = P[0]-np.median(P[0])
    starts = []
    thresh = np.std(P_med)
    #thresh = 0.2
    for k in range(1,P_med.shape[0],1):
        #if (P_med[k]>P_med[k-1]) and (P_med[k-1]==0):
        #if (P_med[k]!=P_med[k-1]):
        if (P_med[k]>thresh):
            starts.append(k)
    
    starts = np.array(starts)/15.*2

    # if False:
        # starts_lockout = []
        # starts_lockout.append(starts[0])
        # for k in range(1, starts.shape[0], 1):
            # #if (abstimes[idx04][k]-abstimes[idx04][k-1])>0.500:
            # if (starts[k]-starts[k-1])>lockout:
                # starts_lockout.append(starts[k])
        
        # starts = np.array(starts_lockout)
    # else:
        # pass
    
    
    # plot feature traces, power-spectrogram peaks, argrelmax peaks and initiation peaks
    if plotting:
        fig=plt.figure()
        #locs = np.array(scipy.signal.argrelmax(P_med)).squeeze()/15.*2
        
        ax=plt.subplot(1,1,1)
        plt.plot(times+shift/15., np.abs(feature), label='Original trace')
        plt.plot(times[::2][:-1],P_med, c='red', label='Power_median')
        #plt.scatter(locs, P_med[np.int32(locs*15/2.)], c='green', label='argrelmax of power median')
        plt.scatter(starts, P_med[np.int32(starts*15/2)]*1.1, c='orange', label='Starts')
        plt.legend(fontsize=20)
        plt.show()

    return starts


def visualize_lever_vs_DLClever(starts_arrays, root_dir):
    
    abstimes, abspositions, abscodes = load_lever_data(root_dir)

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
    shifts = np.arange(-5.0, 10.0, 0.03)
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
    
def visualize_lever_vs_starts(starts_arrays, root_dir, shift = 4.5):
    
    abstimes, abspositions, abscodes = load_lever_data(root_dir)

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
    

def find_quiet_periods_specific_length(starts_arrays, length_quiet_period):

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
    
    return starts[idx]/1E3, durations

#
def find_quiet_periods_specific_length_and_first_feature(starts_arrays,
                                                         length_quiet_period,
                                                         plotting=False):

    # discretize time in miliseconds
    time_bins = np.zeros((1400*1000),'int32')
    starts_arrays = starts_arrays/15.  # CONVERT TO SECONDS

    # bin all activity initiations into time_bins array
    for k in range(len(starts_arrays)):
        if k != 2:
            time_bins[np.int32(starts_arrays[k]*1000)]+=1
        else:
            print ("Excluding nose movement from assessment of overall body movement")
    #
    if plotting:
        ax=plt.subplot(2,1,1)
        ax.plot(np.arange(time_bins.shape[0])/1E3,time_bins)

    # find all quiet periods > 1 time step (i.e. 1 millisecond)
    idx = np.where(time_bins==0)[0]
    print ("idx: ", idx.shape)
    starts = []
    ends = []
    starts.append(idx[0])
    for k in range(1,idx.shape[0],1):
        if (idx[k]-idx[k-1])>1:
            starts.append(k)
            ends.append(k-1)
    #
    if len(starts)>len(ends):
        starts =starts[:-1]
    print ("starts: ", starts)
    #
    starts = np.array(starts)/1E3
    ends = np.array(ends)/1E3

    quiet_periods = np.vstack((starts, ends)).T
    print ("inits: ", quiet_periods.shape, quiet_periods)

    # get durations of quiet
    durations = quiet_periods[:,1]-quiet_periods[:,0]
    print ("durations: ", durations.shape, durations)

    # find periods above suggested lockout
    idx = np.where(durations>length_quiet_period)[0]
    durations = durations[idx]
    quiet_periods = quiet_periods[idx]

    print ("durations > min: ", durations.shape, durations)

    # show the starts of quiet periods
    if plotting:
        ax2=plt.subplot(2,1,2)
        ax2.scatter(starts,durations)

        plt.scatter(starts[idx],durations[idx])
        plt.title("Quiet period length: "+ str(length_quiet_period)+ "sec;    #: "+str(idx.shape[0]), fontsize=20)
        plt.suptitle(os.path.split(fname)[1])
        plt.show()

    print ("# of quiet_periods longer than ", length_quiet_period, "   #", quiet_periods.shape)

    return quiet_periods

def find_quiet_periods_all(starts_arrays):

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
    durations = []
    for k in range(len(starts)):
        durations.append((inside[k][-1]-inside[k][0])/1000.)
    durations = np.array(durations)


    # ax=plt.subplot(2,1,2)
    # plt.scatter(np.array(starts)/1E3,durations)
        # #starts = np.array(starts); ends=np.array(ends)
    # ##print (starts.shape, ends.shape)

    # # find periods above suggested lockout
    # lockout_len = length_quiet_period*1000 # do 6sec lockouts
    # print ("durations")
    # starts = np.array(starts)
    # idx = np.where(durations>length_quiet_period)[0]
    # print (idx.shape)
    # plt.scatter(starts[idx]/1E3,durations[idx])
    # plt.title(" Quiet periods  ", fontsize=15)

    # plt.suptitle("Quiet period length: "+ str(length_quiet_period)+ "sec;    #: "+str(idx.shape[0]), fontsize=20)
    # plt.show()

    # print ("# of quiet_periods longer than ", length_quiet_period, "   #", starts[idx].shape)
    
    return durations



def quiet_periods_histogram(starts_arrays,
                            plotting=False,
                            length_quiet_period=0):

    # discretize time in miliseconds
    time_bins = np.zeros((1400*1000),'int32')
    
    # bin all activity starts;
    for k in range(len(starts_arrays)):
        if k != 2:
            time_bins[np.int32(starts_arrays[k]*1000)]+=1
        else:
            print ("Excluding nose movement from assessment of overall body movement")
    if plotting:
        ax=plt.subplot(2,1,1)
        plt.plot(np.arange(time_bins.shape[0])/1E3,time_bins)

    # find quiet periods and cycle through them
    idx = np.where(time_bins==0)[0]
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
    durations = []
    for k in range(len(starts)):
        durations.append((inside[k][-1]-inside[k][0])/1000.)
    #
    # if plotting:
    #     ax=plt.subplot(2,1,2)
    #     plt.scatter(np.array(starts)/1E3,durations)


    # find periods above suggested lockout
    lockout_len = length_quiet_period*1000 # do 6sec lockouts
    durations = np.array(durations)

    starts = np.array(starts)
    idx = np.where(durations>length_quiet_period)[0]

    # if plotting:
    #  plt.scatter(starts[idx]/1E3,durations[idx])
    #  plt.title(" Quiet periods  ", fontsize=15)
    #
    #  plt.suptitle("Quiet period length: "+ str(length_quiet_period)+ "sec;    #: "+str(idx.shape[0]), fontsize=20)
    #  plt.show()

    print ("# of quiet_periods longer than ", length_quiet_period, "   #", starts[idx].shape)
    
    return starts[idx]/1E3, durations
    
def correlate_lever_dlc(k, temp_norm, traces_upsampled, abspos_upsampled):

    #for k in trange(start_,end_,1, desc='Correlating lever_motor_angle with DLC '+
     #                                labels[correlation_id]):
    temp = np.correlate(traces_upsampled,
                        np.roll(abspos_upsampled,k), mode='valid')
    return temp[0]/temp_norm
