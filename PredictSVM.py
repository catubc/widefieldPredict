import matplotlib

import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import os
import cv2

import glob2

from sklearn.svm import SVC # "Support vector classifier"
import matplotlib.patches as mpatches


class PredictSVM():

    def __init__(self):
        pass

    # generate 80% - 20% training - testing datasets
    def generate_training_data(self, trial_courses_fixed, trial_courses_random_fixed):
        selected_trials = np.arange(int(trial_courses_fixed.shape[0]*0.8)) # of trials to separate the train vs test data
        selected_trials_random = np.arange(int(trial_courses_random_fixed.shape[0]*0.8)) # of trials to separate the train vs test data

        trial_courses_fixed_ids = []
        trial_courses_random_fixed_ids = []
        ctr=0
        while True:
            trial_courses_fixed_ids.append(selected_trials+ctr)
            trial_courses_random_fixed_ids.append(selected_trials_random+ctr)

            ctr+=1
            if (ctr+selected_trials.shape[0]>=trial_courses_fixed.shape[0]):
                break
            if (ctr+selected_trials_random.shape[0]>=trial_courses_random_fixed.shape[0]):
                break

        return trial_courses_fixed_ids, trial_courses_random_fixed_ids

    #
    def generate_training_data_10fold(self, trial_courses_fixed, trial_courses_random_fixed):

        n_trials = int(trial_courses_fixed.shape[0]*0.8)

        #selected_trials = np.arange(int(trial_courses_fixed.shape[0]*0.8)) # of trials to separate the train vs test data
        #selected_trials_random = np.arange(int(trial_courses_random_fixed.shape[0]*0.8)) # of trials to separate the train vs test data

        trial_courses_fixed_ids = []
        trial_courses_random_fixed_ids = []

        for k in range(10):
            trial_courses_fixed_ids.append(np.random.choice(np.arange(trial_courses_fixed.shape[0]),
                                                            n_trials,
                                                           replace=False))
            trial_courses_random_fixed_ids.append(np.random.choice(np.arange(trial_courses_fixed.shape[0]),
                                                            n_trials,
                                                           replace=False))


        return trial_courses_fixed_ids, trial_courses_random_fixed_ids



    def normalize_data(self, data1, data2, random_flag):

        #print ("NIORMALIZEION: ", data1.shape, data2.shape)
        data_in = np.vstack((data1,data2))

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

        return data1, data2

    #
    def make_training_sets_multiple_tests_window2(self,
                                                  time,
                                                 time_window,
                                                 trial_courses_fixed,
                                                 trial_courses_fixed_ids,
                                                 trial_courses_random_fixed,
                                                 trial_courses_random_ids):

        # combine good trials with random trials for training sets:
        good_trials = trial_courses_fixed[trial_courses_fixed_ids, :,time:time+time_window].reshape(trial_courses_fixed_ids.shape[0], -1)
        temp = np.arange(trial_courses_fixed.shape[0])
        idx = np.delete(temp,trial_courses_fixed_ids) # remove the training set trials for testing stage
        test_trials = trial_courses_fixed[idx, :,time:time+time_window].reshape(idx.shape[0], -1)  # select left over trials for testing;

        #
        random_trials = trial_courses_random_fixed[trial_courses_random_ids, :,time:time+time_window].reshape(trial_courses_random_ids.shape[0], -1)
        temp = np.arange(trial_courses_random_fixed.shape[0])
        idx = np.delete(temp,trial_courses_random_ids) # remove training set trials for testing
        test_trials_random = trial_courses_random_fixed[idx, :,time:time+time_window].reshape(idx.shape[0], -1)

        # rebalance the data to ensure SVM doesn't overfit;
        # Cat TODO: this is a big issue in cases where insufficient postiive of negative trials are presnt:
        max_n_trials = min(good_trials.shape[0], random_trials.shape[0])
        good_trials = good_trials[:max_n_trials]
        random_trials = random_trials[:max_n_trials]

        #
        max_n_trials = min(test_trials.shape[0], test_trials_random.shape[0])
        test_trials = test_trials[:max_n_trials]
        test_trials_random = test_trials_random[:max_n_trials]


        # make labels
        y = np.zeros(good_trials.shape[0]+random_trials.shape[0],'int32')
        y[:good_trials.shape[0]]=1

        # concatenate
        X = np.vstack((good_trials,random_trials))

        return X, y, test_trials, test_trials_random



    def parallel_svm_multiple_tests2(self,
                                     time,
                                    time_window,
                                    trial_courses_fixed,
                                    trial_courses_fixed_ids,
                                    trial_courses_random_fixed,
                                    trial_courses_random_ids,
                                    random_flag,
                                    root_dir):


        #
        trial_courses_fixed, trial_courses_random_fixed = \
                    self.normalize_data(trial_courses_fixed, trial_courses_random_fixed, random_flag)

        #
        res1 = []
        res2 = []
        sens = []
        spec = []
        accuracy = []
        for k in range(len(trial_courses_fixed_ids)):
            X, y, test_trials, test_trials_random = self.make_training_sets_multiple_tests_window2(time,
                                                                                             time_window,
                                                                                             trial_courses_fixed,
                                                                                             trial_courses_fixed_ids[k],
                                                                                             trial_courses_random_fixed,
                                                                                             trial_courses_random_ids[k])
            #
            model = SVC(kernel='linear', C=1)
            model.fit(X, y)
            support_vectors = model.support_vectors_

            #
            yfit = model.predict(test_trials)
            true_pos = np.sum(yfit)
            false_neg = (test_trials.shape[0]-true_pos)
            res1.append(np.sum(yfit)/float(yfit.shape[0]))

            # random data test
            yfit = model.predict(test_trials_random)
            false_pos = np.sum(yfit)
            true_neg = (test_trials_random.shape[0]-false_pos)
            res2.append(np.sum(yfit)/float(yfit.shape[0]))

            # compute sensitivity:  true positives / (true positives + false negatives)
            sens.append(true_pos / (true_pos+false_neg))
            spec.append(true_neg / (true_neg+false_pos))

            all_pos = true_pos + false_neg
            all_neg = false_pos + true_neg
            # compute accuracy
            accuracy_temp = (true_pos+true_neg)/(all_pos+all_neg)

            if False:
                print ("")
                print ("SVM run: ", k)
                print ("true_pos :", true_pos)
                print ("true_neg :", true_neg)
                print ("false_neg :", false_neg)
                print ("false_pos :", false_pos)
                print ("all_pos :", all_pos)
                print ("all_neg :", all_neg)
                print ("accuracy :", accuracy_temp)

            # compute accuracy:
            accuracy.append(accuracy_temp)

        #return (res1, res2, sens, spec)
        np.save(root_dir + str(time).zfill(5)+'_sens.npy', sens)
        np.save(root_dir + str(time).zfill(5)+'_spec.npy', spec)
        np.save(root_dir + str(time).zfill(5)+'_accuracy.npy', accuracy)

        return (res1, res2)
    #
    def plot_accuracy2(self,
                       root_dir,
                       length_rec,
                       fig, ax,
                       clr, label_in, labels):
        time_window=1
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

        if True:
            plt.errorbar(t,
                     accuracy_array, accuracy_error, linestyle='None', marker='^', c=clr, alpha=.7)

        if False:
            from scipy.signal import savgol_filter

            yhat = savgol_filter(accuracy_array, 3, 1) #
            plt.plot(t, yhat, c=clr,linewidth=5, alpha=.7)
        else:
            plt.plot(t, accuracy_array, c=clr,linewidth=5, alpha=.7)



        patch_j = mpatches.Patch(color = clr, label = label_in[:-1])
        labels.append(patch_j)

        plt.ylim(0,1)
        plt.xlim(t[0],t[-1])
        #plt.legend(handles=labels)
        plt.ylim(0,1)
        plt.plot([t[0],t[-1]],[0.5,0.5],'r--',c='black')
        plt.plot([0,0],[0,1],'r--',c='black')
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)

        plt.suptitle("Sliding time window prediction using "+str(time_window)+" frames = "+str(round(time_window/30.,2))+"sec", fontsize=20)
        plt.xlabel("Time (sec)",fontsize=20)
        plt.suptitle(root_dir, fontsize=20)
        #plt.show()
        return labels, ax

    def find_sessions(self):
         # load ordered sessions from file
        self.sessions = np.load(os.path.join(self.root_dir, self.animal_id,'tif_files.npy'))
        #print (self.sessions)

        data = []
        for k in range(len(self.sessions)):
            data.append(os.path.split(self.sessions[k])[1][:-4])
        self.sessions = data

        #
        if self.session_id != 'all':
            final_session = []
            for k in range(len(self.sessions)):
                if self.session_id in self.sessions[k]:
                    final_session = [self.sessions[k]]
                    break

            self.sessions = final_session



    def predictSVM(self):

        # find specific session if only 1 chosen
        self.find_sessions()

        #
        prefix1 = ''
        if self.lockout:
            prefix = '_lockout_'+str(self.lockout_window)+"sec_"
        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)


        for s in range(len(self.sessions)):

            fname = os.path.join(self.root_dir, self.animal_id,'tif_files',
                                 self.sessions[s],
                                 self.sessions[s]+'_'+
                                 self.code+
                                 prefix1+
                                 '_trial_ROItimeCourses_'+
                                 str(self.window)+'sec'+
                                 prefix2+
                                 '.npy'
                                 )
            #
            print ("FNAME: ", fname)
            trial_courses_fixed = np.load(fname)
            trial_courses_random_fixed = np.load(fname.replace('trial','random'))

            # cross validation; slides through the data and makes as many 80% batches as possible:
            #  E.g. if 65 trials, 80% is 52, so we can get 14 different batches
            # ratio of testing 80% train to 20% test is fixed inside function
            trial_courses_fixed_ids, trial_courses_random_fixed_ids = \
                                                self.generate_training_data_10fold(trial_courses_fixed,
                                                                              trial_courses_random_fixed)

            # make dir to save SVM data for each time ponit
            root_dir=os.path.split(fname)[0]+'/'
            try:
                os.mkdir(root_dir+'/analysis')
            except:
                pass

            # 
            if True:
                times = np.arange(0,trial_courses_fixed.shape[2])
                if self.parallel:
                    res = parmap.map(self.parallel_svm_multiple_tests2,
                                     times,
                                     self.sliding_window,
                                     trial_courses_fixed,
                                     trial_courses_fixed_ids, # ids of trial sto be used; make sure
                                     trial_courses_random_fixed,
                                     trial_courses_random_fixed_ids,
                                     self.random_flag,
                                     root_dir+'/analysis/',
                                     pm_processes=self.n_cores,
                                     pm_pbar=True)
                else:
                    for k in range(len(times)):
                        self.parallel_svm_multiple_tests2(times[k],
                                     self.sliding_window,
                                     trial_courses_fixed,
                                     trial_courses_fixed_ids, # ids of trial sto be used; make sure
                                     trial_courses_random_fixed,
                                     trial_courses_random_fixed_ids,
                                     self.random_flag,
                                     root_dir+'/analysis/')
                print ("DONE")



            # collect data
            fnames = np.sort(glob2.glob(root_dir+'/analysis/'+"*accuracy.npy"))

            #
            data_out = np.zeros((len(fnames),10),'float32')
            for ctr,fname in enumerate(fnames):
                data_out[ctr]=np.load(fname)

            # make SVM_Scores output for animal
            try:
                os.mkdir(os.path.join(self.root_dir, self.animal_id, "SVM_Scores"))
            except:
                pass
            
            # make fname out for animal + session
            fname_out = os.path.join(self.root_dir, self.animal_id,
                                 'SVM_Scores',
                                 'SVM_Scores_'+
                                 self.sessions[s]+'_'+
                                 self.code+
                                 prefix1+
                                 '_trial_ROItimeCourses_'+
                                 str(self.window)+'sec'+
                                 prefix2+
                                 '.npy'
                                 )

            np.save(fname_out,data_out)
