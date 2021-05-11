import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import os
import glob2
import warnings
import pickle

import sklearn
from sklearn.svm import SVC # "Support vector classifier"
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.pyplot import MultipleLocator

from tqdm import trange, tqdm

class PredictSVMTime():

    def __init__(self):

        self.code = 'code_04'


    def get_sessions(self):
         # load ordered sessions from file
        self.sessions = np.load(os.path.join(self.main_dir,
                                             self.animal_id,
                                             'tif_files.npy'))
        # grab session name only
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

        # fix binary string files issues; remove 'b and ' from file names
        for k in range(len(self.sessions)):
            self.sessions[k] = str(self.sessions[k]).replace("'b",'').replace("'","")
            if self.sessions[k][0]=='b':
                self.sessions[k] = self.sessions[k][1:]

        self.sessions = np.array(self.sessions)


    def get_fname(self): # load ordered sessions from file

        self.sessions = np.load(os.path.join(self.main_dir, self.animal_id,'tif_files.npy'))

        data = []
        for k in range(len(self.sessions)):
            data.append(os.path.split(self.sessions[k])[1][:-4])
        self.sessions = data

        #
        final_session = []
        for k in range(len(self.sessions)):
            if str(self.session_id) in str(self.sessions[k]):

                final_session = self.sessions[k]
                break

        self.session = final_session

        print ("self session: ", self.session)

        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        # make fname out for animal + session
        fname = os.path.join(self.main_dir, self.animal_id,'tif_files',
                     self.session,
                     self.session+'_'+
                     self.code+
                     prefix1+
                     '_trial_ROItimeCourses_'+
                     str(self.window)+'sec'+
                     prefix2+
                     '.npy'
                     )

        self.fname = fname
        self.fname_random = fname.replace('trial','random')


    def predict(self):

        # make sure out dir exists
        try:
            os.mkdir(os.path.join(self.main_dir, self.animal_id, "SVM_Times"))
        except:
            pass

        self.get_sessions()
        #
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)


        for s in trange(len(self.sessions)):
            # make fname out for animal + session
            self.session_id = self.sessions[s]
            self.fname = os.path.join(self.main_dir, self.animal_id,'tif_files',
                                 self.session_id,
                                 self.session_id+'_'+
                                 self.code+
                                 prefix1+
                                 '_trial_ROItimeCourses_'+
                                 str(self.window)+'sec'+
                                 prefix2+
                                 '.npy'
                                 )
            #
            self.process_data2()

    #classification of time (10-class)
    def assign_class(self, X_assign):
        X_l=X_assign[:,:300,:]
        X_k=X_l[:,0:30,:]
        X_k=X_k.reshape(X_k.shape[0],X_k.shape[1]*X_k.shape[2])
        for i in range(30,271,30):
            X_t = X_l[:,i:i+30,:]
            X_t=X_t.reshape(X_t.shape[0],X_t.shape[1]*X_t.shape[2])
            X_k=np.concatenate((X_k,X_t),axis=0)

        y_ct=np.zeros(X_assign.shape[0])
        for i in range(1,10):
            Z_ct=i*np.ones(X_assign.shape[0])
            y_ct=np.concatenate((y_ct,Z_ct))

        X_tSVM=X_k
        y_tSVM=y_ct
        return X_tSVM,y_tSVM


    def process_data2(self):

        warnings.filterwarnings("ignore")
        try:
            data_leverpress = np.load(self.fname)
        except:
            print ("no such filename, skipping")
            return

        X=data_leverpress #
        self.n_trials = X.shape[0]

        #
        if self.n_trials<10:
            print (" too few trials... skipping")
            return

        # switch last 2 dimensions; UNCLEAR WHY NEEDED...
        X=X.transpose((0,2,1))

        #normalize
        X_R= X.reshape(-1,X.shape[1]*X.shape[2])
        normal_X = preprocessing.normalize(X_R)
        n_X=normal_X.reshape(X.shape[0],X.shape[1],X.shape[2])
        X=n_X

        X_nonnan=X[~np.isnan(X)]
        X_lever=X_nonnan.reshape((X.shape[0],X.shape[1],-1))


        #10-fold confusion matrix
        clf = svm.SVC() # Non-linear classifier
        ten_svm=[]
        conf_matrix_ten=[]
        kf = KFold(n_splits=10,random_state=None, shuffle=True)
        for train_index, test_index in kf.split(X_lever):
          X_train_assign, X_test_assign = X_lever[train_index], X_lever[test_index]

          # assign training
          X_train_k,y_train_k=self.assign_class(X_train_assign)

          # assign testing
          X_test_k,y_test_k=self.assign_class(X_test_assign)

          #
          clf.fit(X_train_k, y_train_k)
          score=clf.score(X_test_k, y_test_k)
          y_predicted=clf.predict(X_test_k)

          #
          cm=confusion_matrix(y_test_k,y_predicted)
          confusion_m=cm.T # make each row be the prediction
          conf_matrix_norm = confusion_m.astype('float') / confusion_m.sum(axis=1)[:,np.newaxis] #calculate the precision
          conf_matrix_norm = np.nan_to_num(conf_matrix_norm)
          ten_svm.append(score)
          conf_matrix_ten.append(conf_matrix_norm)

        #
        self.get_fname_out()

        # save data
        np.savez(self.fname_out,
                 conf_matrix_ten = conf_matrix_ten,
                 n_trials = self.n_trials)


    def get_fname_out(self):
        #
        dir_out = os.path.join(self.main_dir, self.animal_id,
                                 "SVM_Times")
        #
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        #
        self.fname_out = os.path.join(dir_out,
                     "SVM_Times_"+
                     self.session_id+'_'+
                     self.code+
                     prefix1+
                     '_trial_ROItimeCourses_'+
                     str(self.window)+'sec'+
                     prefix2
                     )

    def plot_decision_time(self, ax=None):

        #
        if ax is None:
            ax=plt.subplot(111)
        clrs = ['black','blue']
        names = ['all', 'lockout']
        #
        lockouts = [False, True]
        ctr=0
        self.n_trials_both = []
        for lockout in lockouts:
            self.lockout = lockout
            self.get_fname_out()

            #
            data = np.load(self.fname_out+'.npz')
            self.conf_matrix_ten = data['conf_matrix_ten']
            self.n_trials_both.append(data['n_trials'])

            #
            sc=np.mean(self.conf_matrix_ten,axis=0) # mean of confusion matrix

            #
            confusion_10=sc.diagonal()
            confusion_s=self.conf_matrix_ten

            confusion_d=[]
            for i in range(0,10):
                confusion_d.append(confusion_s[i].diagonal())
            x_std=np.std(confusion_d,axis=0) #/(10**0.5)

            t = np.arange(-9.5,0.5,1)
            ax.plot(t,confusion_10, c=clrs[ctr])

            ax.fill_between(t, confusion_10+x_std,
                               confusion_10-x_std,
                               color=clrs[ctr],
                              alpha=.2,
                    label=names[ctr])
            ctr+=1
        ax.set_ylim(0,1.0)
        ax.set_xlim(-10,t[-1])

        ax.plot([t[0],t[-1]], [0.1,0.1], 'r--')


class PredictSVMChoice():

    def __init__(self):

        self.min_trials = 0
        print ("Set min trials to : ", self.min_trials)


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

    #
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
        # check sizes; sometimes there are 1 or 2 less random courses
        if trial_courses_random_fixed.shape[0]!=trial_courses_fixed.shape[0]:
            # pad the random courses with a random example shifted by half of time
            idx = np.random.choice(np.arange(trial_courses_random_fixed.shape[0]),1)
            rolled = np.roll(trial_courses_random_fixed[idx],trial_courses_random_fixed[idx].shape[2]//2,axis=1)
            trial_courses_random_fixed = np.vstack((rolled, trial_courses_random_fixed))




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
        # Cat TODO: this is a big issue in cases where insufficient postiive of negative trials are present:
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
            print ("X input to SVC", X.shape, "  y input to SVC: ", y.shape)
            model = SVC(kernel='linear', C=1)
            model.fit(X, y)
            # support_vectors = model.support_vectors_

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

            # compute accuracy:
            accuracy.append(accuracy_temp)

        #return (res1, res2, sens, spec)
        np.save(root_dir + str(time).zfill(5)+'_sens.npy', sens)
        np.save(root_dir + str(time).zfill(5)+'_spec.npy', spec)
        np.save(root_dir + str(time).zfill(5)+'_accuracy.npy', accuracy)

        # save the last model to disk
        filename = root_dir + str(time).zfill(5)+'_svm.pkl'
        pickle.dump(model, open(filename, 'wb'))

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

    def get_sessions(self):
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

        # fix binary string files issues; remove 'b and ' from file names
        for k in range(len(self.sessions)):
            self.sessions[k] = str(self.sessions[k]).replace("'b",'').replace("'","")
            if self.sessions[k][0]=='b':
                self.sessions[k] = self.sessions[k][1:]

    def predict(self):

        # find specific session if only 1 chosen
        self.get_sessions()

        #
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)


        for s in range(len(self.sessions)):
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

            if os.path.exists(fname_out) and self.overwrite==False:
                print ("   already computed, skipping ...")
                continue

            # grab trial and random data
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
            try:
                trial_courses_fixed = np.load(fname)
                trial_courses_random_fixed = np.load(fname.replace('trial','random'))
            except:
                print (" ....file not found, skippping ")
                continue

            # cross validation; slides through the data and makes as many 80% batches as possible:
            #  E.g. if 65 trials, 80% is 52, so we can get 14 different batches
            # ratio of testing 80% train to 20% test is fixed inside function
            if trial_courses_fixed.shape[0] < self.min_trials:
                print ("    Skipping too few trials less than ", self.min_trials)
                continue

            #
            print ("processing; ", fname, " with lockout: ", self.lockout)

            #
            trial_courses_fixed_ids, trial_courses_random_fixed_ids = \
                                                self.generate_training_data_10fold(trial_courses_fixed,
                                                                              trial_courses_random_fixed)

            # make dir to save SVM data for each time ponit
            root_dir=os.path.split(fname)[0]+'/'
            try:
                os.mkdir(root_dir+'/analysis')
            except:
                pass

            # exclude small # of trial data
            if trial_courses_fixed.shape[0]<=1:
                print ("  Insuffciient trials, exiting...")
                continue

            #
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

            np.save(fname_out,data_out)

        print ("DONE predicting SVM on animal: ", self.animal_id)


    def predict2(self):
        ''' Predict2 is an updated version which uses sklearn tools for svm instead of
        coding from scratch

        '''


        # make SVM_Scores output for animal
        try:
            os.mkdir(os.path.join(self.root_dir, self.animal_id, "SVM_Scores"))
        except:
            pass


        # find specific session if only 1 chosen
        self.get_sessions()

        #
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        for s in range(len(self.sessions)):

            # make SVM_Scores output for session
            # try:
            #     os.mkdir(os.path.join(self.root_dir,
            #                           self.animal_id,
            #                           "SVM_Scores",
            #                           self.sessions[s]))
            # except:
            #     pass


            # make fname out for animal + session
            fname_out = os.path.join(self.root_dir,
                                     self.animal_id,
                                     'SVM_Scores',
                                     self.sessions[s],
                                     'SVM_Scores_'+
                                     self.sessions[s]+'_'+
                                     self.code+
                                     prefix1+
                                     '_trial_ROItimeCourses_'+
                                     str(self.window)+'sec'+
                                     prefix2+
                                     "_Xvalid"+str(self.cross_validation)+
                                     "_Slidewindow"+str(self.sliding_window)+
                                     '.npy'
                                     )

            if os.path.exists(fname_out) and self.overwrite==False:
                print ("   already computed, skipping ...")
                continue

            # grab trial and random data
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
            try:
                trial_courses_fixed = np.load(fname)
                trial_courses_random_fixed = np.load(fname.replace('trial','random'))
            except:
                print (" ....file not found, skippping ")
                continue

            if trial_courses_fixed.shape[0] < self.min_trials:
                print ("    Skipping too few trials less than ", self.min_trials)
                continue

            #
            print ("processing; ", fname, " with lockout: ", self.lockout)

            # exclude small # of trial data
            if trial_courses_fixed.shape[0]<=1:
                print ("  Insuffciient trials, exiting...")
                continue


            # cross validation; slides through the data and makes as many 80% batches as possible:
            #  E.g. if 65 trials, 80% is 52, so we can get 14 different batches
            # ratio of testing 80% train to 20% test is fixed inside function
            #data_split = 0.8

            #
            self.trials = trial_courses_fixed
            self.random = trial_courses_random_fixed
            accuracy, labels, predictions = self.compute_accuracy_svm_KFold()

            #
            fname_out = os.path.join(self.root_dir, self.animal_id,'SVM_Scores',
                                 'SVM_Scores_'+
                                 self.sessions[s]+
                                 self.code+
                                 prefix1+
                                 '_trial_ROItimeCourses_'+
                                 str(self.window)+'sec'+
                                 "_Xvalid"+str(self.xvalidation)+
                                 "_Slidewindow"+str(self.sliding_window)+
                                 '.npz'
                                 )

            # # grab trial and random data
            # fname = os.path.join(self.root_dir, self.animal_id,'SVM_Scores',
            #                      self.sessions[s],
            #                      self.sessions[s]+'_'+
            #                      self.code+
            #                      prefix1+
            #                      '_trial_ROItimeCourses_'+
            #                      str(self.window)+'sec'+
            #                      prefix2+
            #                      '.npy'
            #                      )

            np.savez(fname_out,
                    accuracy = accuracy,
                    labels = labels,
                    predictions = predictions)



            #
            #np.save(fname_out,accuracy)

        print ("DONE predicting SVM on animal: ", self.animal_id)



    #
    def compute_accuracy_svm_KFold(self):

        # randomize seed
        np.random.seed()

        # select groups for parallel processing
        run_ids = np.arange(self.xvalidation)

        idx_trials_split = np.array_split(np.random.choice(np.arange(self.trials.shape[0]),
                                                           self.trials.shape[0],
                                                           replace=False),
                                         self.xvalidation)

        idx_random_split = np.array_split(np.random.choice(np.arange(self.random.shape[0]),
                                                           self.random.shape[0],
                                                           replace=False),
                                         self.xvalidation)

        data = parmap.map(run_svm_single_randomized_kFold,
                           run_ids,
                           idx_trials_split,
                           idx_random_split,
                           self.trials,
                           self.random,
                           self.sliding_window,
                           self.method,
                           pm_processes = self.n_cores,
                           pm_pbar=True)

        #
        accuracy = []
        labels = []
        predictions = []
        for k in range(len(data)):
            accuracy.append(data[k][0].T)
            labels.append(data[k][1].T)
            predictions.append(data[k][2].T)

        accuracy = np.vstack(accuracy).T
        labels = np.vstack(labels).T
        predictions = np.vstack(predictions).T


        return accuracy, labels, predictions

    #
    def compute_accuracy_svm(self,
                             trials,
                             random,
                             data_split):

        # shuffle data x times
        run_ids = np.arange(self.cross_validation)

        #
        accuracy = parmap.map(run_svm_single_randomized,
                              run_ids,
                              self.sliding_window,
                              trials,
                              random,
                              data_split,
                              pm_processes = self.n_cores)

        accuracy = np.array(accuracy)

        return accuracy


def run_svm_single_randomized(run_id,
                              window,
                              trials,
                              random,
                              data_split):

    #
    np.random.seed()

    # shuffle data and draw random samples of the same maount
    idx_trials = np.random.choice(np.arange(trials.shape[0]),
                           int(trials.shape[0]*data_split),
                           replace=False)
    idx_random = np.random.choice(np.arange(random.shape[0]),
                           int(trials.shape[0]*data_split),
                           replace=False)

    # stack data
    train = np.vstack((trials[idx_trials],random[idx_random]))
    labels_train = np.hstack((np.ones(trials[idx_trials].shape[0]),
                       np.zeros(random[idx_random].shape[0])))

    #
    idx_trials_not = np.delete(np.arange(trials.shape[0]),idx_trials)
    idx_random_not = np.delete(np.arange(random.shape[0]),idx_random)
    test = np.vstack((trials[idx_trials_not], random[idx_random_not]))
    labels_test = np.hstack((np.ones(trials[idx_trials_not].shape[0]),
                       np.zeros(random[idx_random_not].shape[0])))

    #
    accuracy_local=[]
    for k in range(0,trials.shape[2]-window,1):
        X = train#[:,:,:window]
        X = X[:,:,k:k+window]
        # if mean_filter:
        #     X = np.mean(X,2)

        X = X.reshape(train.shape[0],-1)

        #
        y = labels_train

        #
        X = sklearn.preprocessing.scale(X)

        #
        clf = svm.SVC(kernel='rbf')
        clf.fit(X, y)


        # test
        X_test = test[:,:,k:k+window]
        # if mean_filter:
        #     X_test = np.mean(X_test,2)

        X_test = X_test.reshape(X_test.shape[0],-1)


        X_test = sklearn.preprocessing.scale(X_test)
        #
        y_pred = clf.predict(X_test)

        #
        acc = accuracy_score(labels_test, y_pred)
        accuracy_local.append(acc)

    return accuracy_local



def run_svm_single_randomized_kFold(run_id,
                                   idx_trials_split,
                                   idx_random_split,
                                   trials,
                                   random,
                                   sliding_window,
                                   method):

    # train data excludes the run_id
    idx_trials = np.delete(np.arange(trials.shape[0]),
                               idx_trials_split[run_id])
    idx_random = np.delete(np.arange(random.shape[0]),
                               idx_random_split[run_id])

    # test data is the left over labels
    idx_trials_not = np.delete(np.arange(trials.shape[0]),idx_trials)
    idx_random_not = np.delete(np.arange(random.shape[0]),idx_random)

    # stack train data
    train = np.vstack((trials[idx_trials],random[idx_random]))
    labels_train = np.hstack((np.ones(trials[idx_trials].shape[0]),
                              np.zeros(random[idx_random].shape[0])))

    # stack test data
    test = np.vstack((trials[idx_trials_not], random[idx_random_not]))
    labels_test = np.hstack((np.ones(trials[idx_trials_not].shape[0]),
                             np.zeros(random[idx_random_not].shape[0])))

    #
    accuracy2=[]
    labels2 = []
    pred2 = []
    for k in range(0, trials.shape[2]-sliding_window, 1):
        X = train#[:,:,:window]
        X = X[:,:,k:k+sliding_window]
        #if mean_filter:
        #    X = np.mean(X,2)

        X = X.reshape(train.shape[0],-1)

        #
        y = labels_train

        #
        X = sklearn.preprocessing.scale(X)

        #
        clf = svm.SVC(kernel=method)
        clf.fit(X, y)


        # test
        X_test = test[:,:,k:k+sliding_window]

        X_test = X_test.reshape(X_test.shape[0],-1)
        X_test = sklearn.preprocessing.scale(X_test)
        #
        y_pred = clf.predict(X_test)

        #
        acc = accuracy_score(labels_test, y_pred)
        accuracy2.append(acc)
        labels2.append(labels_test)
        pred2.append(y_pred)

    accuracy2 = np.array(accuracy2)
    labels2 = np.array(labels2)
    pred2 = np.array(pred2)

    #print ("inner loop: accraucy: ", accuracy2.shape, labels2.shape, pred2.shape)
    return accuracy2, labels2, pred2
