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
import torch
import pickle as pk

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC # "Support vector classifier"
import matplotlib.patches as mpatches
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.pyplot import MultipleLocator
from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split
from sklearn import datasets

from tqdm import trange, tqdm

class PredictMultiState():

    def __init__(self):

        self.labels = ['left_paw',
                        'right_paw',
                        'nose',
                        'jaw',
                        'right_ear',
                        'tongue',
                        'lever',
                        'all',
                        'code_04',
                        'code_04_lockout']



    def load_ca_whole_stack(self, times):

        #
        data = np.load(self.fname_Ca_time_filters)
        #if self.verbose:
        #    print (" whole stack: ", data.shape)

        #
        stack = []
        for t in times:
            temp = data[int(t-self.window)*self.imaging_rate:int(t+self.window)*self.imaging_rate]
            if temp.shape[0]==self.window*self.imaging_rate*2:
                stack.append(temp)
        temp = np.array(stack)

        temp = temp.transpose(0,2,1)
        #if self.verbose:
        #    print ("  Ca stack: ", temp.shape)

        return temp

    def load_data(self, session):

        # data = np.load('')
        # iris = datasets.load_iris()#Store variables as target y and the first two features as X (sepal length and sepal width of the iris flowers)

        # grab movement initiation arrays
        fname = os.path.join(self.root_dir, self.animal_id,'tif_files',
                             session,
                             session+'_'+
                             str(self.no_movement)+"secNoMove_movements.npz"
                             )

        #
        data = np.load(fname, allow_pickle=True)
        self.feature_quiescent = data['feature_quiescent']
        self.all_quiescent = data['all_quiescent']

        self.trial_times = []
        for k in range(len(self.feature_quiescent)):
            print (self.labels[k], np.array(self.feature_quiescent[k]).shape)

            temp_trials = np.vstack(self.feature_quiescent[self.selected_feature_id])
            self.trial_times.append(temp_trials[:,1])

    def pre_svm_run(self):
        # load [Ca] data
        self.trials = self.load_ca_whole_stack(self.trial_times)

        # get random times for that feature
        self.generate_random_trials()

        # load [Ca] data
        self.random = self.load_ca_whole_stack(self.random_times)

        # ensure same size
        n_max_trials = min(self.trials.shape[0], self.random.shape[0])
        self.trials = self.trials[:n_max_trials]
        self.random = self.random[:n_max_trials]

        # check to see if below min size for xvalidation
        if n_max_trials<self.min_trials:
            self.accuracy_array.append([])
            self.prediction_array.append([])
            self.n_trials_array.append(n_max_trials)
            return


        # self.data = ...
        # self.labels = ...


    def set_data(self):

        self.X = self.data
        self.y = self.labels

    def prepare_and_run_svm(self):

        print ("Trials stack (make sure same across D1): ", self.trials.shape)
        for k in range(len(self.trials)):
            print (k, " size: ", self.trials[k].shape)

        # select window
        t_array = np.arange(0, self.trials[0].shape[1], self.sliding_window_step)

        if self.parallel:
            self.acc = parmap.map(run_classifier, t_array,
                                  self.window,
                                  self.trials,
                                  self.classifier,
                                  pm_pbar=True,
                                  pm_processes = self.n_cores
                                  )
        else:
            self.acc = []
            for t in t_array:
                self.acc.append(run_classifier(t,
                                self.window,
                                self.trials,
                                self.classifier))



def run_classifier(t,
                   window,
                   trials,
                   classifier,
                   ):

    #
    t1 = t
    t2 = t + window

    trials_window = trials[:,:,t1:t2]
    #print (trials_window.shape)

    trials_window_flat = trials_window.reshape(trials_window.shape[0],
                                               trials_window.shape[1],
                                               -1)
    #print (trials_window_flat.shape)

    # 10-fold split; should we randomize?
    idx = np.array_split(np.random.choice(np.arange(trials_window_flat.shape[1]),
                                          trials_window_flat.shape[1], replace=False),
                         10)
    #
    acc = []
    for k in range(10):
        acc.append(run_10fold(k, idx, trials_window_flat, classifier))

    return acc

    # run 10-fold prediction;
    # can parallelize this loop

def run_10fold(k,
               idx,
               trials_window_flat,
               classifier,
               ):

    #
    idx_test = idx[k]
    X_test = trials_window_flat[:,idx_test]
    y_test = []
    for f in range(X_test.shape[0]):
        y_test.append(np.zeros(X_test[f].shape[0])+f)
    y_test = np.hstack(y_test)
    X_test = X_test.reshape(-1, X_test.shape[2])
    #print ("X_test: ", X_test.shape, " y_test: ", y_test.shape)

    #
    idx_train = np.delete(np.arange(trials_window_flat.shape[1]), idx[k])
    train = trials_window_flat[:,idx_train]
    # print ("Train: ", train.shape)

    # loop over all features/body parts and generate labels
    y_train = []
    X_train = []
    for f in range(train.shape[0]):
        y_train.append(np.zeros(train[f].shape[0])+f)
        X_train.append(train[f])

    X_train = np.vstack(X_train)
    y_train = np.hstack(y_train)
    #print ("X_train: ", X_train.shape, " y_train: ", y_train.shape)

    # STANDARDIZE DATA
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #
    # Fit SVM/Classifier

    if classifier == 'svm':
        acc = run_svm_multi_variate(X_train, y_train, X_test, y_test)
    elif classifier == 'random_forest':
        acc = run_random_forest_multi_variate(X_train, y_train, X_test, y_test)
    else:
        print ("Classifer uknonwn")
        return None
    return acc

# X_train, X_test, y_train, y_test =

def run_svm_multi_variate(X_train, y_train, X_test, y_test):

    #linear = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo').fit(X_train, y_train)
    #rbf = svm.SVC(kernel='rbf', gamma=1, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    #poly = svm.SVC(kernel='poly', degree=3, C=1, decision_function_shape='ovo').fit(X_train, y_train)
    sig = svm.SVC(kernel='sigmoid', C=1, decision_function_shape='ovo').fit(X_train, y_train)

    # retrieve the accuracy and print it for all 4 kernel functions
    #accuracy_lin = linear.score(X_test, y_test)
    #accuracy_poly = poly.score(X_test, y_test)
    #accuracy_rbf = rbf.score(X_test, y_test)
    accuracy_sig = sig.score(X_test, y_test)

    #
    return accuracy_sig


def run_random_forest_multi_variate(X_train, y_train, X_test, y_test):

    #  Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=10,
                                        n_jobs=1)
    classifier.fit(X_train, y_train)

    # # Predicting the Test set results
    # y_pred = classifier.predict(X_test)
    #
    # #Reverse factorize (converting y_pred from 0s,1s and 2s to Iris-setosa, Iris-versicolor and Iris-virginica
    # reversefactor = dict(zip(range(3),definitions))
    # y_test = np.vectorize(reversefactor.get)(y_test)
    # y_pred = np.vectorize(reversefactor.get)(y_pred)

    accuracy_sig = classifier.score(X_test, y_test)

    return accuracy_sig

class PredictSVMTime():

    def __init__(self):

        self.code = 'code_04'

        #
        self.imaging_rate = 30

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
        kf = KFold(n_splits=10,
                   random_state=None,
                   shuffle=True)
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

        #
        #self.min_trials = 0
        #print ("Set min trials to : ", self.min_trials)

        #
        self.imaging_rate = 30

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
    #
    # def predict(self):
    #
    #     # find specific session if only 1 chosen
    #     self.get_sessions()
    #
    #     #
    #     prefix1 = ''
    #     if self.lockout:
    #         prefix1 = '_lockout_'+str(self.lockout_window)+"sec"
    #
    #     #
    #     prefix2 = ''
    #     if self.pca_flag:
    #         prefix2 = '_pca_'+str(self.pca_var)
    #
    #
    #     for s in range(len(self.sessions)):
    #         # make fname out for animal + session
    #         fname_out = os.path.join(self.root_dir, self.animal_id,
    #                              'SVM_Scores',
    #                              'SVM_Scores_'+
    #                              self.sessions[s]+'_'+
    #                              self.code+
    #                              prefix1+
    #                              '_trial_ROItimeCourses_'+
    #                              str(self.window)+'sec'+
    #                              prefix2+
    #                              '.npy'
    #                              )
    #
    #         if os.path.exists(fname_out) and self.overwrite==False:
    #             print ("   already computed, skipping ...")
    #             continue
    #
    #         # grab trial and random data
    #         fname = os.path.join(self.root_dir, self.animal_id,'tif_files',
    #                              self.sessions[s],
    #                              self.sessions[s]+'_'+
    #                              self.code+
    #                              prefix1+
    #                              '_trial_ROItimeCourses_'+
    #                              str(self.window)+'sec'+
    #                              prefix2+
    #                              '.npy'
    #                              )
    #         try:
    #             trial_courses_fixed = np.load(fname)
    #             trial_courses_random_fixed = np.load(fname.replace('trial','random'))
    #         except:
    #             print (" ....file not found, skippping ")
    #             continue
    #
    #         # cross validation; slides through the data and makes as many 80% batches as possible:
    #         #  E.g. if 65 trials, 80% is 52, so we can get 14 different batches
    #         # ratio of testing 80% train to 20% test is fixed inside function
    #         if trial_courses_fixed.shape[0] < self.min_trials:
    #             print ("    Skipping too few trials less than ", self.min_trials)
    #             continue
    #
    #         #
    #         print ("processing; ", fname, " with lockout: ", self.lockout)
    #
    #         #
    #         trial_courses_fixed_ids, trial_courses_random_fixed_ids = \
    #                                             self.generate_training_data_10fold(trial_courses_fixed,
    #                                                                           trial_courses_random_fixed)
    #
    #         # make dir to save SVM data for each time ponit
    #         root_dir=os.path.split(fname)[0]+'/'
    #         try:
    #             os.mkdir(root_dir+'/analysis')
    #         except:
    #             pass
    #
    #         # exclude small # of trial data
    #         if trial_courses_fixed.shape[0]<=1:
    #             print ("  Insuffciient trials, exiting...")
    #             continue
    #
    #         #
    #         times = np.arange(0,trial_courses_fixed.shape[2])
    #         if self.parallel:
    #             res = parmap.map(self.parallel_svm_multiple_tests2,
    #                              times,
    #                              self.sliding_window,
    #                              trial_courses_fixed,
    #                              trial_courses_fixed_ids, # ids of trial sto be used; make sure
    #                              trial_courses_random_fixed,
    #                              trial_courses_random_fixed_ids,
    #                              self.random_flag,
    #                              root_dir+'/analysis/',
    #                              pm_processes=self.n_cores,
    #                              pm_pbar=True)
    #         else:
    #             for k in range(len(times)):
    #                 self.parallel_svm_multiple_tests2(times[k],
    #                              self.sliding_window,
    #                              trial_courses_fixed,
    #                              trial_courses_fixed_ids, # ids of trial sto be used; make sure
    #                              trial_courses_random_fixed,
    #                              trial_courses_random_fixed_ids,
    #                              self.random_flag,
    #                              root_dir+'/analysis/')
    #
    #         # collect data
    #         fnames = np.sort(glob2.glob(root_dir+'/analysis/'+"*accuracy.npy"))
    #
    #         #
    #         data_out = np.zeros((len(fnames),10),'float32')
    #         for ctr,fname in enumerate(fnames):
    #             data_out[ctr]=np.load(fname)
    #
    #         # make SVM_Scores output for animal
    #         try:
    #             os.mkdir(os.path.join(self.root_dir, self.animal_id, "SVM_Scores"))
    #         except:
    #             pass
    #
    #         np.save(fname_out,data_out)
    #
    #     print ("DONE predicting SVM on animal: ", self.animal_id)

    def generate_random_trials(self):

        # generate random times outside the window of real trials
        random = []
        for k in range(10000):
            t = np.random.rand()*1250+self.window*2
            if np.min(np.abs(t-self.trial_times))>= self.random_lockout:
                random.append(t)

            if len(random)==len(self.trial_times):
                break
        #
        self.random_times = np.array(random)

    #
    def load_ca_whole_stack(self, times):

        #
        data = np.load(self.fname_Ca_time_filters)
        #if self.verbose:
        #    print (" whole stack: ", data.shape)

        #
        stack = []
        for t in times:
            temp = data[int(t-self.window)*self.imaging_rate:int(t+self.window)*self.imaging_rate]
            if temp.shape[0]==self.window*self.imaging_rate*2:
                stack.append(temp)
        temp = np.array(stack)

        temp = temp.transpose(0,2,1)
        #if self.verbose:
        #    print ("  Ca stack: ", temp.shape)

        return temp


    def load_code04_times(self,recording):

        #
        root_dir = self.root_dir
        lockout_window = self.lockout_window

        #
        try:
            fname = os.path.join(root_dir,self.animal_id, 'tif_files',recording,
                                 recording + '_locs44threshold.npy')
            locs_44threshold = np.load(fname)
        except:
            print ("locs 44 thrshold missing", recording)
            self.locs_code04 = np.zeros((0),'float32')
            self.locs_code04_lockout = np.zeros((0),'float32')
            return

        #
        codes = np.load(os.path.join(root_dir,self.animal_id, 'tif_files',recording,
                                 recording + '_code44threshold.npy'))
        code = b'04'
        idx = np.where(codes==code)[0]
        locs_selected = locs_44threshold[idx]

        if locs_selected.shape[0]==0:
            self.locs_code04 = np.zeros((0),'float32')
            self.locs_code04_lockout = np.zeros((0),'float32')
            return

        diffs = locs_selected[1:]-locs_selected[:-1]
        idx = np.where(diffs>lockout_window)[0]

        #
        locs_selected_with_lockout = locs_selected[idx+1]
        if locs_selected_with_lockout.shape[0]==0:
            self.locs_code04 = np.zeros((0),'float32')
            self.locs_code04_lockout = np.zeros((0),'float32')
            return

        # ADD FIRST VAL
        if locs_selected[0]>lockout_window:
            locs_selected_with_lockout = np.concatenate(([locs_selected[0]], locs_selected_with_lockout), axis=0)

        self.locs_code04 = locs_selected
        self.locs_code04_lockout = locs_selected_with_lockout

    def pre_svm_run(self):
        # load [Ca] data
        self.trials = self.load_ca_whole_stack(self.trial_times)

        # get random times for that feature
        self.generate_random_trials()

        # load [Ca] data
        self.random = self.load_ca_whole_stack(self.random_times)

        # ensure same size
        n_max_trials = min(self.trials.shape[0], self.random.shape[0])
        self.trials = self.trials[:n_max_trials]
        self.random = self.random[:n_max_trials]

        # check to see if below min size for xvalidation
        if n_max_trials<self.min_trials:
            self.accuracy_array.append([])
            self.prediction_array.append([])
            self.n_trials_array.append(n_max_trials)
            return

        # run svm
        accuracy, labels, predictions = self.compute_accuracy_svm_KFold()

        # append data
        self.accuracy_array.append(accuracy)
        self.prediction_array.append(predictions)
        self.n_trials_array.append(n_max_trials)

    def predict_whole_stack(self):

        # make SVM_Scores output for animal
        try:
            os.mkdir(os.path.join(self.root_dir, self.animal_id, "SVM_Scores"))
        except:
            pass

        # load all sessions or select only chosen one
        self.get_sessions()

        #
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        #
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca'+str(self.nComp)+"Components_"

        for s in range(len(self.sessions)):
            #
            print ("  running: ", self.sessions[s])

            #  name of accuracy pickle file to save at the end.
            fname_out = os.path.join(self.root_dir, self.animal_id,'SVM_Scores',
                                 'SVM_Scores_'+
                                 self.sessions[s]+
                                 self.code+
                                 prefix1+
                                 prefix2+
                                 #'_trial_ROItimeCourses_'+
                                 'window'+str(self.window)+"sec"+
                                 "_Xvalid"+str(self.xvalidation)+
                                 "_Slidewindow"+str(self.sliding_window)+"Frames"+
                                 '_accuracy.pk'
                                 )

            # filename of the PCA denoised whole stack
            self.fname_Ca_time_filters = os.path.join(self.root_dir, self.animal_id,'tif_files',
                                                     self.sessions[s],
                                                     self.sessions[s]+"_whole_stack_trial_ROItimeCourses_"+
                                                     str(self.window)+'sec'+
                                                     '_pca'+str(self.all_comps)+'components.npy'
                                                     )


            if os.path.exists(self.fname_Ca_time_filters)==False:
                print ("   [Ca] file missing, skipping ...")
                continue
            #
            if os.path.exists(fname_out) and self.overwrite==False:
                print ("   already computed, skipping ...")
                continue

            # get body initiations from the DLC movement file:
            self.load_trial_times_whole_stack(self.sessions[s])

            # get code_04 rewarded lever pulls from raw data
            self.load_code04_times(self.sessions[s])

            ##################################################################
            ############## COMPUTE SVM FOR BODY MOVEMENTS ####################
            ##################################################################
            self.accuracy_array = []
            self.prediction_array = []
            self.n_trials_array = []

            for k in range(len(self.feature_quiescent)):
                #
                if len(self.feature_quiescent[k])<self.min_trials:
                    self.accuracy_array.append([])
                    self.prediction_array.append([])
                    self.n_trials_array.append(len(self.feature_quiescent[k]))
                    continue

                temp_trials = np.vstack(self.feature_quiescent[k])
                self.trial_times = temp_trials[:,1]
                self.pre_svm_run()

            ##################################################################
            ############## COMPUTE SVM FOR QUIESCENCE PERIODS ONLY ###########
            ##################################################################
            if len(self.all_quiescent)<self.min_trials:
                self.accuracy_array.append([])
                self.prediction_array.append([])
                self.n_trials_array.append(len(self.all_quiescent))
            else:
                temp_trials = np.vstack(self.all_quiescent)
                self.trial_times = temp_trials[:,1]
                self.pre_svm_run()

            ###############################################
            ########### COMPUTE SVM FOR CODE_04 ###########
            ###############################################
            self.trial_times = self.locs_code04
            if len(self.trial_times)<self.min_trials:
                self.accuracy_array.append([])
                self.prediction_array.append([])
                self.n_trials_array.append(len(self.trial_times))
            else:
                self.pre_svm_run()

            ############################################################
            ########### COMPUTE SVM FOR CODE_04 WITH LOCKOUT ###########
            ############################################################
            self.trial_times = self.locs_code04_lockout
            if len(self.trial_times)<self.min_trials:
                self.accuracy_array.append([])
                self.prediction_array.append([])
                self.n_trials_array.append(len(self.trial_times))
            else:
                self.pre_svm_run()

            ################################################
            #
            with open(fname_out, 'wb') as fout:
                fout.write(pickle.dumps(self.accuracy_array))
            with open(fname_out.replace('accuracy','predictions'), 'wb') as fout:
                fout.write(pickle.dumps(self.prediction_array))

        print ("DONE predicting SVM on animal: ", self.animal_id)

    #
    def load_trial_times_whole_stack(self, session):

        # grab movement initiation arrays
        fname = os.path.join(self.root_dir, self.animal_id,'tif_files',
                             session,
                             session+'_'+
                             str(self.no_movement)+"secNoMove_movements.npz"
                             )

        # if no file return empty arrays?
        if os.path.exists(fname)==False:
            self.feature_quiescent = []
            self.all_quiescent = []
            #
            for k in range(7):
                self.feature_quiescent.append([])
            return
        #
        data = np.load(fname, allow_pickle=True)
        self.feature_quiescent = data['feature_quiescent']
        self.all_quiescent = data['all_quiescent']

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

            #
            self.trials = trial_courses_fixed
            self.random = trial_courses_random_fixed
            accuracy, labels, predictions = self.compute_accuracy_svm_KFold()

            #
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
        if self.parallel:
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
        else:
            data = []
            for k in range(run_ids.shape[0]):
                #print ("self.trials.shape: ", self.trials.shape)
                data.append(run_svm_single_randomized_kFold(
                                                           run_ids[k],
                                                           idx_trials_split,
                                                           idx_random_split,
                                                           self.trials,
                                                           self.random,
                                                           self.sliding_window,
                                                           self.method)
                            )

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
# def run_svm_single_randomized(run_id,
#                               window,
#                               trials,
#                               random,
#                               data_split):
#
#     #
#     np.random.seed()
#
#     # shuffle data and draw random samples of the same maount
#     idx_trials = np.random.choice(np.arange(trials.shape[0]),
#                            int(trials.shape[0]*data_split),
#                            replace=False)
#     idx_random = np.random.choice(np.arange(random.shape[0]),
#                            int(trials.shape[0]*data_split),
#                            replace=False)
#
#     # stack data
#     train = np.vstack((trials[idx_trials],random[idx_random]))
#     labels_train = np.hstack((np.ones(trials[idx_trials].shape[0]),
#                        np.zeros(random[idx_random].shape[0])))
#
#     #
#     idx_trials_not = np.delete(np.arange(trials.shape[0]),idx_trials)
#     idx_random_not = np.delete(np.arange(random.shape[0]),idx_random)
#     test = np.vstack((trials[idx_trials_not], random[idx_random_not]))
#     labels_test = np.hstack((np.ones(trials[idx_trials_not].shape[0]),
#                        np.zeros(random[idx_random_not].shape[0])))
#
#     #
#     accuracy_local=[]
#     for k in range(0,trials.shape[2]-window,1):
#         X = train#[:,:,:window]
#         X = X[:,:,k:k+window]
#         # if mean_filter:
#         #     X = np.mean(X,2)
#
#         X = X.reshape(train.shape[0],-1)
#
#         #
#         y = labels_train
#
#         #
#         X = sklearn.preprocessing.scale(X)
#
#         #
#         clf = svm.SVC(kernel='rbf')
#         clf.fit(X, y)
#
#
#         # test
#         X_test = test[:,:,k:k+window]
#         # if mean_filter:
#         #     X_test = np.mean(X_test,2)
#
#         X_test = X_test.reshape(X_test.shape[0],-1)
#
#
#         X_test = sklearn.preprocessing.scale(X_test)
#         #
#         y_pred = clf.predict(X_test)
#
#         #
#         acc = accuracy_score(labels_test, y_pred)
#         accuracy_local.append(acc)
#
#     return accuracy_local


class PredictSVMConcatenated():

    def __init__(self):
        pass

    def expand(self, r1, r2):
        r1 = torch.from_numpy(r1.transpose(0,2,1))
        r2 = torch.from_numpy(r2)
        r3 = torch.matmul(r1, r2)

        del r1
        del r2
        torch.cuda.empty_cache()

        return r3

    def concatenate_denoised_sessions(self):

        #
        trials = []
        random = []
        trials_in_sess = []
        for session in self.session_selected:
            fname = os.path.join(self.root_dir, self.animal_id,
                                 'tif_files',
                                 session,
                                 session+"_globalPcaDenoised.npz")
            data = np.load(fname, allow_pickle=True)

            t = data['trials_time_filters']
            r = data['random_time_filters']

            # clip number of trials in case too few random sessions.
            max_trials = min(t.shape[0],r.shape[0])
            t = t[:max_trials]
            r = r[:max_trials]

            trials.append(t)
            random.append(r)
            trials_in_sess.append(t.shape[0])

        self.trials = np.vstack(trials)
        self.random = np.vstack(random)
        self.trials_in_sess = trials_in_sess

        if self.verbose:
            print (self.trials.shape, self.random.shape)


    def denoise_sessions(self,
                         subsample=False):

        #
        for session in self.session_selected:
            root_dir = self.root_dir+self.animal_id + '/tif_files/'
            a, b = self.load_trial(root_dir, session, subsample, self.pca_var)

            t1, t2 = self.pca_denoise_data(self.pca, a, self.nComp)
            r1, r2 = self.pca_denoise_data(self.pca, b, self.nComp)

            fname_out = os.path.join(self.root_dir, self.animal_id, 'tif_files', session,
                                    session+"_globalPcaDenoised.npz")

            np.savez(fname_out,
                    trials_time_filters = t1,
                    trials_space_filters = t2,
                    random_time_filters = r1,
                    random_space_filters = r2,
                    sessions = self.session_selected,
                    nComp = self.nComp
                    )

    def load_trial(self, root_dir, session, subsample,
                  pca_val):

        r1 = np.load(root_dir+session+"/"+session+"_code_04_random_ROItimeCourses_30sec_pca_"+
                     str(pca_val)+".npy")[:,:,:900]
        r2 = np.load(root_dir+session+"/"+session+"_code_04_random_ROItimeCourses_30sec_pca_"+
                     str(pca_val)+"_spatial.npy")
        if self.verbose:
            print ("r1; ", r1.shape, '  r2: ', r2.shape)


        if subsample:
            idx = np.random.choice(np.arange(r1.shape[0]),
                                   10,
                                   replace=False)
            r1 = r1[idx]

        r3 = self.expand(r1,r2)
        if self.verbose:
            print (r1.shape, " to ", r3.shape)

        t1 = np.load(root_dir+session+"/"+session+"_code_04_trial_ROItimeCourses_30sec_pca_"+
                     str(pca_val)+".npy")[:,:,:900]
        t2 = np.load(root_dir+session+"/"+session+"_code_04_trial_ROItimeCourses_30sec_pca_"+
                     str(pca_val)+"_spatial.npy")

        #
        if subsample:
            t1 = t1[idx]
        t3 = self.expand(t1,t2)

        return r3, t3


    def make_pca_object(self):

        #

        X = self.r.reshape(self.r.shape[0]*self.r.shape[1],
                                     self.r.shape[2])
        if self.verbose:
            print (" data in pre reshape:", self.r.shape)
            print (" data for pca: ", X.shape)

        # subselect data
        n_selected = max(2500, int(X.shape[0]*self.pca_global_min_frames))
        n_selected = min(X.shape[0], n_selected)


        #
        idx = np.random.choice(np.arange(X.shape[0]),n_selected,
                               replace=False)
        X_select = X[idx]

        #
        pca = PCA()
        if self.verbose:
            print (" data subsampled for pca: ", X_select.shape)
        #
        pca.fit(X_select)

        fname_pca = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                 self.session_selected[0],
                                 self.session_selected[0]+
                                 "_globalPca_min_trials_concatenated"+
                                 str(self.min_trials_concatenated)+
                                 '_code_04_'+
                                 str(self.window)+'sec'+
                                 '.pkl'
                                 )

        #print ("fname_pca: ", fname_pca)
        pk.dump(pca, open(fname_pca,"wb"))

        # compute # of components needed for reconsturction to the requierd limit
        expl_variance = pca.explained_variance_
        #
        expl_variance = expl_variance/expl_variance.sum(0)
        if self.verbose:
            print ("Normalized variance explained (first 5 PCs): ",
                   expl_variance[:5])

        sums = 0
        pca_explained_var_val = 0.99
        for k in range(expl_variance.shape[0]):
            sums+=expl_variance[k]
            if sums>=pca_explained_var_val:
                nComp = k+1
                break

        #
        if self.verbose:
            print ("nComp required: for var: ", pca_explained_var_val, " is: ",  k)
            print ("manually set nComp to ", 20)
        self.pca = pca
        self.nComp = 20


    def check_complete(self):

        #
        fname_out = os.path.join(self.root_dir,self.animal_id,'tif_files',
                 self.session_selected[0],
                 self.session_selected[0]+
                 "_globalPca_min_trials_concatenated"+
                 str(self.min_trials_concatenated)+
                 '_code_04_'+
                 str(self.window)+'sec'+
                 '_accuracy.npz'
                         )

        # save the concatenated info metadata
        np.save(fname_out[:-4]+"_sessions.npy",
                self.session_selected)
        np.save(fname_out[:-4]+"_n_trials.npy",
                self.n_trials_selected)

        #
        if os.path.exists(fname_out):
            if self.verbose:
                print ("File already exists: ", fname_out)
            return True

        print ("... processing: ", self.session_selected[0])
        return False

    #
    def compute_accuracy_svm_concatenated(self):

        if self.verbose:
            print (" running compute_accuracy_svm")

        # randomize seed
        np.random.seed()

        # split data by taking each session's length into account;
        idx_trials_split = []
        idx_random_split = []
        for p in range(self.xvalidation):
            idx_trials_split.append([])
            idx_random_split.append([])

        #
        count = 0
        for k in range(len(self.trials_in_sess)):
            idx_split_local = np.array_split(np.random.choice(
                                                np.arange(self.trials_in_sess[k]),
                                                           self.trials_in_sess[k],
                                                           replace=False),
                                                           self.xvalidation)

            idx_random_local = np.array_split(np.random.choice(
                                                np.arange(self.trials_in_sess[k]),
                                                           self.trials_in_sess[k],
                                                           replace=False),
                                                           self.xvalidation)

            # add selected ids from each session individually to ensure equall
            #   submsampling of each session rather than the overall concatenated stack
            for p in range(self.xvalidation):
                idx_trials_split[p].extend(idx_split_local[p]+count)
                idx_random_split[p].extend(idx_random_local[p]+count)

            # advnace count:
            count+=self.trials_in_sess[k]

        # select groups for parallel processing
        run_ids = np.arange(self.xvalidation)
        if self.parallel:
            data = parmap.map(run_svm_single_randomized_kFold,
                                           run_ids,
                                           idx_trials_split,
                                           idx_random_split,
                                           self.trials,
                                           self.random,
                                           self.sliding_window,
                                           self.method,                                           pm_processes = self.xvalidation,
                                           pm_pbar=False)

        else:
            data = []
            for k in range(run_ids.shape[0]):
                print ("   running xvaldiation: ", k,
                       "  self.trials.shape: ", self.trials.shape)
                data.append(run_svm_single_randomized_kFold(
                                                           run_ids[k],
                                                           idx_trials_split,
                                                           idx_random_split,
                                                           self.trials,
                                                           self.random,
                                                           self.sliding_window,
                                                           self.method))


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

        #
        fname_out = os.path.join(self.root_dir,self.animal_id,'tif_files',
                         self.session_selected[0],
                         self.session_selected[0]+
                         "_globalPca_min_trials_concatenated"+
                         str(self.min_trials_concatenated)+
                         '_code_04_'+
                         str(self.window)+'sec'+
                         '_accuracy.npz'
                         )

        np.savez(fname_out,
                accuracy = accuracy,
                labels = labels,
                predictions = predictions)


    def pca_denoise_data(self, pca, data_stm, nComp):

        #
        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2])

        #
        time_filters = pca.transform(X)[:,:nComp]

        #
        pca_time_filters = time_filters.reshape(data_stm.shape[0],
                                                data_stm.shape[1],
                                                -1).transpose(0,2,1)
        pca_spatial_filters = pca.components_[:nComp,:]

        if self.verbose:
            print ("... made data: ", pca_time_filters.shape)

        return pca_time_filters, pca_spatial_filters

    # find no. of trials per session:
    def get_no_trials(self):

        #
        fnames= np.load(self.root_dir + self.animal_id+'/tif_files.npy')
        f = []
        for fname in fnames:
            f.append(os.path.split(fname)[-1][:-4])
        fnames=np.vstack(f)

        # get # of trial in each session
        n_trials = []
        sess_id = []
        for fname in fnames:
            t = self.root_dir+self.animal_id+"/tif_files/"+fname[0]+"/"+fname[0]+"_all_locs_selected.txt"

            t2 = self.root_dir+self.animal_id+"/tif_files/"+fname[0]+"/"+fname[0]+"_"+str(self.code)+"_trial_ROItimeCourses_"+str(self.window)+"sec_pca_"+str(self.pca_var)+".npy"
            #
            try:
                #d = np.loadtxt(t,
                #               dtype='str')
                d2 = np.load(t2)
                sess_id.append(fname[0])
                n_trials.append(d2.shape[0])
            except:
                pass

        self.n_trials = n_trials
        self.sess_id = sess_id

        if self.verbose:
            print (self.sess_id[:5])
            print (self.n_trials[:5])

        #return n_trials, sess_id

    def concatenate_trials(self):

        print ('')
        print ('')
        sess_con = []
        n_tr_con = []
        for k in range(len(self.sess_id)):

            if self.n_trials[k]>=self.min_trials_in_single_session:
                total_tr = self.n_trials[k]
                sess_local = []
                sess_local.append(self.sess_id[k])

                #
                if total_tr< self.min_trials_concatenated:
                    # loop forward
                    for p in range(k+1, len(self.sess_id),1):
                        if self.n_trials[p]>=self.min_trials_in_single_session:
                            total_tr+= self.n_trials[p]
                            sess_local.append(self.sess_id[p])
                            if total_tr >= self.min_trials_concatenated:
                                break

                sess_con.append(sess_local)
                n_tr_con.append(total_tr)

        self.sessions_con = sess_con
        self.n_trials_con = n_tr_con

        ctr=0
        for k in range(len(self.sessions_con)):
            print (self.sessions_con[k], self.n_trials_con[k])
            print ("")
            ctr+=1
        print (ctr)

#print (svm.n_trials_con)

    def compute_pca_data(self):

        #
        subsample= True #take only 10 trials per session for computing PCA object initially

        # sliding window loop over data:
        r = np.zeros((0,900,16384),'float32')  # Use extra 30 frame window?!?!
        for session in self.session_selected:
            root_dir = self.root_dir+self.animal_id+'/tif_files/'

            #
            a, b = self.load_trial(root_dir, session, subsample, self.pca_var)
            r = np.vstack((r,a))
            r = np.vstack((r,b))

            if self.verbose:
                print ("done session: ", session)

        #
        if self.verbose:
            print ("Total stack: ", r.shape)
        self.r = r


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

        # normalization
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

#
#
# # single standing function that can be parallelized
# def run_svm_single_randomized_kFold(
#                                    run_id,
#                                    idx_trials_split,
#                                    idx_random_split,
#                                    window,
#                                    method,
#                                    trials,
#                                    random):
#
#     # train data excludes the run_id
#     idx_trials = np.delete(np.arange(trials.shape[0]),
#                                idx_trials_split[run_id])
#     idx_random = np.delete(np.arange(random.shape[0]),
#                                idx_random_split[run_id])
#
#     # test data is the left over labels
#     idx_trials_not = np.delete(np.arange(trials.shape[0]),idx_trials)
#     idx_random_not = np.delete(np.arange(random.shape[0]),idx_random)
#
#     # stack train data
#     train = np.vstack((trials[idx_trials],random[idx_random]))
#     labels_train = np.hstack((np.ones(trials[idx_trials].shape[0]),
#                               np.zeros(random[idx_random].shape[0])))
#
#     # stack test data
#     test = np.vstack((trials[idx_trials_not], random[idx_random_not]))
#     labels_test = np.hstack((np.ones(trials[idx_trials_not].shape[0]),
#                              np.zeros(random[idx_random_not].shape[0])))
#
#     #
#     accuracy2=[]
#     labels2 = []
#     pred2 = []
#     for k in range(0, trials.shape[2]-window, 1):
#         X = train#[:,:,:window]
#         X = X[:,:,k:k+window]
#         #if mean_filter:
#         #    X = np.mean(X,2)
#
#         X = X.reshape(train.shape[0],-1)
#
#         #
#         y = labels_train
#
#         #
#         X = sklearn.preprocessing.scale(X)
#
#         #
#         clf = svm.SVC(kernel=method)
#         clf.fit(X, y)
#
#
#         # test
#         X_test = test[:,:,k:k+window]
#
#         X_test = X_test.reshape(X_test.shape[0],-1)
#         X_test = sklearn.preprocessing.scale(X_test)
#         #
#         y_pred = clf.predict(X_test)
#
#         #
#         acc = accuracy_score(labels_test, y_pred)
#         accuracy2.append(acc)
#         labels2.append(labels_test)
#         pred2.append(y_pred)
#
#     accuracy2 = np.array(accuracy2)
#     labels2 = np.array(labels2)
#     pred2 = np.array(pred2)
#
#     #print ("inner loop: accraucy: ", accuracy2.shape, labels2.shape, pred2.shape)
#     return accuracy2, labels2, pred2
