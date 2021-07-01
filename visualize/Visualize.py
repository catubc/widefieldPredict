import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull
from tqdm import trange
from scipy.spatial import cKDTree

import glob2
import scipy
from tqdm import trange
from statsmodels.stats.multitest import multipletests
from scipy.optimize import curve_fit

from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Visualize():

    def __init__(self):

        self.clr_ctr = 0

        #
        self.animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        #
        self.labels = ["M1", "M2", "M3", "M4","M5",'M6']

        #
        self.colors = ['black','blue','red','green','magenta','pink','cyan']

        #
        self.feature_codes = [
                    'left_paw',          # 0
                    'right_paw',         # 1
                    'nose',              # 2
                    'jaw',               # 3
                    'right_ear',         # 4
                    'tongue',            # 5
                    # 'lever',           # 6
                    # 'quiescence',      # 7
                    # 'code_04',         # 8
                    # 'code_04_lockout'  # 9
                    ]

        #
        self.n_colors = [10,19,23,17,20,48]

        #
        self.linewidth = 4

        #
        self.filter=False

        #
        self.cbar_offset = 0

        #
        self.imaging_rate = 30.


    def load_data(self, fname):

        self.data = np.load(fname)



    def format_plot(self, ax):
        ''' Formats plots for decision choice with 50% and 0 lines
        '''
       # meta data
        try:
            xlims = [self.xlim[0],self.xlim[1]]
        except:
            xlims = [-self.window+1,0]

        ylims = [0.4,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.5,0.5],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(0.4,1.0)



    def format_plot2(self, ax):
        ''' Format plots for time prediction
        '''
       # meta data
        xlims = [-10,0]
        ylims = [0.0,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.1,0.1],
                 '--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
       # plt.legend(fontsize=20)


    def get_fname(self): # load ordered sessions from file

        data = np.load(os.path.join(self.main_dir, self.animal_id,'tif_files.npy'))

        #
        self.sessions = []
        for k in range(len(data)):
            self.sessions.append(os.path.split(data[k])[1].replace('.tif',''))

        #
        self.session = None
        for k in range(len(self.sessions)):
            if str(self.session_id) in str(self.sessions[k]):
                self.session = self.sessions[k]
                break

        #
        if self.session is None:
            print (" COULDN't FIND SESSION...")
            self.fname = None
            return


        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        if self.pickle==False:
            #
            fname = os.path.join(self.main_dir, self.animal_id,
                         'SVM_Scores',
                         'SVM_Scores_'+
                         self.session+#"_"+
                         self.code+
                         prefix1+
                         '_trial_ROItimeCourses_'+
                         str(self.window)+'sec_'+
                         "Xvalid"+str(self.xvalidation)+
                         "_Slidewindow"+str(self.sliding_window)+
                         #prefix2+
                         '.npz'
                         )
        else:
            # /media/cat/4TBSSD/yuki/IA1/SVM_Scores/SVM_Scores_IA1pm_Feb1_30Hz_whole_stack_pca30Components_window15sec_Xvalid10_Slidewindow30Frames_accuracy.pk
            fname = os.path.join(self.main_dir, self.animal_id,'SVM_Scores',
                     'SVM_Scores_'+
                     self.session+
                     '_whole_stack_pca30Components_'+
                     'window'+str(self.window)+"sec"+
                     "_Xvalid"+str(self.xvalidation)+
                     "_Slidewindow"+str(self.sliding_window)+"Frames"+
                     '_accuracy.pk'
                     )
        print ("SET FNAME: ", fname)
        self.fname = fname

        # convert wild card file name into correct filename for animal
        main_dir = os.path.join(self.main_dir,
                                self.animal_id,
                                'tif_files')
        session_corrected = os.path.split(
                            glob2.glob(main_dir+"/*"+self.session_id+"*")[0])[1]

        self.session_corrected = session_corrected



    def get_number_of_trials(self):

        ''' There are 4 types of trials so must load them individually
        '''

        # convert wild card file name into correct filename for animal
        main_dir = os.path.join(self.main_dir,
                                self.animal_id,
                                'tif_files')
        session_corrected = os.path.split(
                            glob2.glob(main_dir+"/*"+self.session_id+"*")[0])[1]

        self.session_corrected = session_corrected
        #
        if self.code == 'code_04':

            # check to see if session done
            fname_txt = os.path.join(self.main_dir,
                                     self.animal_id,
                                     'tif_files',
                                     # self.session_id,
                                     # self.session_id+
                                     session_corrected,
                                     session_corrected+"_all_locs_selected.txt")
            try:
                self.n_trials = np.loadtxt(fname_txt).shape[0]
            except:
                self.n_trials = 0
        #
        elif self.code == 'code_04_lockout':
            fname_txt = os.path.join(self.main_dir,
                                     self.animal_id,
                                     'tif_files',
                                     # self.session_id,
                                     # self.session_id+
                                     session_corrected,
                                     session_corrected+"_lockout_10sec_locs_selected.txt")
            try:
                self.n_trials = np.loadtxt(fname_txt).shape[0]
            except:
                self.n_trials = 0
        #
        else:
            fname_data = os.path.join(self.main_dir,
                         self.animal_id,
                         'tif_files',
                         session_corrected,
                         session_corrected+"_3secNoMove_movements.npz")

            data = np.load(fname_data, allow_pickle=True)

            if self.code=='quiescence':
                self.n_trials = data['all_quiescent'].shape[0]
            else:

                # find the code id
                match_code = None
                for p in range(len(self.feature_codes)):
                    if self.code in self.feature_codes[p]:
                        match_code = p
                        break

                self.n_trials = len(data['feature_quiescent'][match_code])




    def get_lever_offset(self):

        fname_lever_offset = os.path.join(self.main_dir,
                                         self.animal_id,
                                         'tif_files',
                                         self.session_corrected,
                                         self.session_corrected+"_lever_offset_n_frames.txt")

        if os.path.exists(fname_lever_offset)==False:

            images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

            aligned_images = np.load(images_file)

            # Find blue light on/off
            blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
            start_blue = 0; end_blue = aligned_images.shape[0]

            if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
                for k in range(len(aligned_images)):
                    if np.average(aligned_images[k])< blue_light_threshold:
                        #self.aligned_images = self.aligned_images[k:]
                        end_blue = k
                        break
            else:                                                           #Case #2: start with light off; remove starting and end chunks;
                #Find first light on
                for k in range(len(aligned_images)):
                    if np.average(aligned_images[k])> blue_light_threshold:
                        start_blue = k
                        break

                #Find light off - count backwards from end of imaging data
                for k in range(len(aligned_images)-1,0,-1):
                    if np.average(aligned_images[k])> blue_light_threshold:
                        end_blue= k
                        break

            self.lever_offset = start_blue

            np.savetxt(fname_lever_offset, [self.lever_offset])

        else:
            self.lever_offset = int(np.loadtxt(fname_lever_offset))

    def plot_decision_choice_all(self):

        #self.get_number_of_trials()


        #
        sessions = np.load(os.path.join(self.main_dir,
                                        self.animal_id,
                                        'tif_files.npy'))

        n_row = int(sessions.shape[0]/10.)+1
        n_row = 7
        n_col = 10

        #
        fnames_pca = glob2.glob(os.path.join(self.main_dir,
                                             self.animal_id,
                                             "SVM_Scores/*.npy"))

        #
        ctr=0
        plt_flag = True
        for session in sessions:
            self.session_id = os.path.split(session)[1][:-4]

            if plt_flag:
                ax=plt.subplot(n_row,n_col,ctr+1)
                ax.set_xticks([])
                ax.set_yticks([])

            # track if session has has some plotting done
            plt_flag = False
            for fname in fnames_pca:
                if self.session_id in str(fname):

                    self.get_number_of_trials()
                    if self.n_trials < self.min_trials:
                        continue

                    if "lockout" in str(fname):
                        self.lockout = True
                        self.plot_decision_choice('blue',
                                                 str(self.pca_var),
                                                 ax)
                        plt_flag = True
                    else:
                        self.lockout = False
                        self.plot_decision_choice('black',
                                                 str(self.pca_var),
                                                 ax)
                        plt_flag = True

            if plt_flag:
                ctr+=1

        #
        plt.suptitle("ANIMAL: "+ self.animal_id +
                     ", Smoothing window: "+str(self.smooth_window)+
                     ", Min # trials: "+str(self.min_trials), fontsize=20)
        plt.show()


    def plot_first_significant(self):

        #
        sessions = np.load(os.path.join(self.main_dir,
                                        self.animal_id,
                                        'tif_files.npy'))
        #
        fnames_pca = glob2.glob(os.path.join(self.main_dir,
                                             self.animal_id,
                                             "SVM_Scores/*.npy"))

        #
        ctr=0
        for session in sessions:
            self.session_id = os.path.split(session)[1][:-4]

            for fname in fnames_pca:
                if self.session_id in str(fname):

                    self.get_number_of_trials()
                    if self.n_trials < self.min_trials:
                        continue

                    if "lockout" in str(fname):
                        self.lockout = True
                        self.process_session()

                        # compute significance
                        self.compute_significance()
                        print (self.sig.shape)

                        # find first significant point in time



                    else:
                        self.lockout = False
                        self.process_session()


    #
    def process_session(self):

        #################################################
        ########### LOAD ACCURACY DATA ##################
        #################################################
        if self.pickle==False:
            try:
                data = np.load(self.fname, allow_pickle=True)
            except:
                print( " ... data missing", self.fname)
                self.data = np.zeros((0))
                return
            self.data = data['accuracy']

        # else load specific code data from file
        else:
            try:
                with (open(self.fname, "rb")) as openfile:
                    data = pickle.load(openfile)
                print ("PICKLE DATA: ", len(data))
            except:
                print( " ... data missing", self.fname)
                self.data = np.zeros((0))
                return

            # find the code id
            match_code = None
            print ("self.code: ", self.code)

            for p in range(len(self.feature_codes)):
                if self.code in self.feature_codes[p]:
                    match_code = p
                    break
            print ("Match code: ", match_code)
            self.data = np.array(data[match_code])

        if self.data.shape[0]==0:
            print ("COULDNT FIND DATA")
            return

        print ("DATA; ", self.data.shape)

        #################################################
        ### LOAD SHIFTS BETWEEN LEVER AND CALCIUM #######
        #################################################
        self.get_lever_offset()
        print ("loaded lever offset: ", self.lever_offset)


        #################################################
        ######## LOAD SHIFTS FROM CORRELATION ANALYSIS ##
        #################################################
        if self.shift_flag:
            fname_correlate = os.path.join(self.main_dir, self.animal_id,
                                 'tif_files', self.session,
                                'correlate.npz')

            try:
                data = np.load(fname_correlate,allow_pickle=True)
            except:
                print( " ... data missing", self.fname)
                self.data = np.zeros((0))
                return

            cors = data['cors'].squeeze().T

            #vis.shift = 0
            print ("SELF SHIFT ID: ", self.shift_id_str)
            if len(self.shift_id_str)>1:
                self.shift_id = int(self.shift_id_str[0])
                self.shift_additional = float(self.shift_id_str[1:])
            else:
                self.shift_id = int(self.shift_id_str)
                self.shift_additional = 0

            print ( " using shift: ", self.shift_id+self.shift_additional)

            corr_featur_id = self.shift_id

            temp_trace = cors[:,corr_featur_id]
            temp_trace[:2000] = 0
            temp_trace[-2000:] = 0
            self.shift = round(np.argmax(temp_trace)/1000. - 15.,2)+self.shift_additional
            print ("SHIFT Loaded: ", self.shift)

        #
        if 'code_04' not in self.code:
            print ("... DLC shift applied: ", self.shift)
            self.data = np.roll(self.data,int(self.shift*self.imaging_rate),axis=0)
        else:
            print ("... DLC shift applied: ", 0)

        #
        print ("... LEVER OFFSET applied: ", self.lever_offset)
        self.data = np.roll(self.data,-self.lever_offset,axis=0)
        print (" rolled data: ", self.data.shape)

        #################################################
        ########### LOAD # TRIALS #######################
        #################################################
        self.get_number_of_trials()
        print (" post trials data size ", self.data.shape)
        print ("      n-trials: ", self.n_trials)


        #
        if self.n_trials<self.min_trials:
            print ("Insufficient trials...", self.n_trials)
            self.data = np.zeros((0))
            return

        #
        mean = self.data.mean(1)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

            mean = self.data.mean(1)

        #
        self.mean = mean
        print (self.mean.shape)

        #
        self.std = np.std(self.data, axis=1)


        # clip the data to the required values
        self.data = self.data[(self.xlim[0]+self.window)*30:
                              (self.xlim[1]+self.window)*30]

        self.mean = self.mean[(self.xlim[0]+self.window)*30:
                              (self.xlim[1]+self.window)*30]
        print ("self mean: ", self.mean.shape)

        self.std = self.std[(self.xlim[0]+self.window)*30:
                              (self.xlim[1]+self.window)*30]


    def process_session_concatenated(self):

        #
        try:
            data = np.load(self.fname, allow_pickle=True)
        except:
            print( " ... data missing", self.fname)
            self.data = np.zeros((0))
            return

        self.data = data['accuracy']

        # grab only first half:
        print ("LOADED: DATA" , self.data.shape)

        # get n trials for both lockout and all trials data
        if False:
            self.get_number_of_trials()
            #print (" post trials data size ", self.data.shape)

            #
            if self.n_trials<self.min_trials:
                print ("Insufficient trials...", self.n_trials)
                self.data = np.zeros((0))
                return

        fname_n_trials = self.fname[:-4]+'_n_trials.npy'
        self.n_trials = np.load(fname_n_trials)

        # gets the corect filename to be loaded below
        self.get_fname()
        #print (" post fname: ", self.data.shape)

        #
        if os.path.exists(self.fname)==False:
            print ("missing: ", self.fname)
            self.data = np.zeros((0))
            return

        #
        mean = self.data.mean(1)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

            mean = self.data.mean(1)

        #
        self.mean = mean

        #
        self.std = np.std(self.data, axis=1)


    def plot_decision_choice(self, clr, label, ax=None):

        #
        self.process_session()

        # get times
        t = np.linspace(-9.5, 9.5, self.mean.shape[0])

        # plotting steps
        if ax is None:
            ax=plt.subplot(111)

        ax.set_title(self.session_id, fontsize=6.5,pad=0.9)
        ax.set_ylabel(str(self.n_trials)+" ("+str(self.n_trials_lockout)+")", fontsize=8)

        #
        ax.plot(t,
                self.mean,
                c=clr,
                label = label,
                linewidth=4)

        ax.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        self.format_plot(ax)

    def compute_significance(self):

        print ("self.data: ", self.data.shape)

        #
        sig = []
        for k in range(self.data.shape[0]):
            #res = stats.ks_2samp(self.data[k],
            #                     control)
            #res = stats.ttest_ind(first, second, axis=0, equal_var=True)

            #
            res = scipy.stats.ttest_1samp(self.data[k], 0.5)

            sig.append(res[1])


        self.sig_save = np.array(sig).copy()
        print ("Self sig save: ", self.sig_save.shape)

        # multiple hypothesis test Benjamini-Hockberg
        temp = np.array(sig)
        print ("data into multi-hypothesis tes:", temp.shape)
        temp2 = multipletests(temp, alpha=self.significance, method='fdr_bh')
        sig = temp2[1]

        # Expand this so we can plot it as a 1D image.
        sig=np.array(sig)[None]

        #
        thresh = self.significance
        idx = np.where(sig>thresh)
        sig[idx] = np.nan

        #
        idx = np.where(self.mean<0.5)
        sig[:,idx] = np.nan

        # save it
        self.sig = sig
        print ("Final sig: ", self.sig.shape)

        # find earliest significant;
        earliest_continuous = 0
        for k in range(self.sig.shape[1]-1,0,-1):
            if np.isnan(self.sig[0][k]):
                earliest_continuous = k+1
                break

            if self.sig[0][k]<=self.significance:
                earliest_continuous = k+1
            else:
                break

        #
        print ("earliest: ", earliest_continuous,
               " in sec: ", -(self.sig.shape[1]-earliest_continuous)/30.)

        self.earliest_continuous = -(self.sig.shape[1]-earliest_continuous)/30.

        self.edt = -(self.sig.shape[1]-earliest_continuous)/30.

        print (" signianc n-trials: ", self.n_trials)

    def compute_first_decoding_time(self, lockouts=[False, True]):

        #
        #if lockouts = [False, True]
        for lockout in lockouts:
            self.lockout=lockout

            all_res_continuous = []
            all_res_earliest = []
            all_session_nos = []
            all_session_names = []
            all_n_trials = []
            all_sigs = []

            #
            for a in trange(len(self.animal_ids)):
                res_continuous = []
                res_earliest = []
                session_nos = []
                session_names = []
                n_trials = []
                sigs = []

                #
                self.animal_id = self.animal_ids[a]

                #
                self.get_sessions()
                #
                for p in range(len(self.session_ids)):
                    self.session_id = self.session_ids[p]

                    #
                    self.fname = self.fnames_svm[p]

                    self.process_session()
                    # print ("a: ", a, self.session_id, self.data.shape)
                    #
                    if self.data.shape[0] == 0:
                        continue

                    # compute significance
                    self.compute_significance()

                    # save all the significant vals;
                    sigs.append(self.sig_save)

                    #
                    self.sig = self.sig.squeeze()

                    # find earliest period of significance, going back in time;
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            break

                    #
                    temp = -self.window+k/self.imaging_rate

                    # Exclude one of the weird datapoint from the AQ2? session
                    if temp>0:
                        #print ("n trials: ", self.n_trials, a,
                        #       p, temp, self.session_id, self.sig.shape)
                        continue

                    #
                    res_continuous.append(temp)

                    # find aboslute earliest
                    k_earliest = self.sig.shape[0]
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            k_earliest = k
                    res_earliest.append(-self.window+k_earliest/self.imaging_rate)

                    #
                    session_nos.append(p)

                    #
                    session_names.append(self.session_id)

                    #
                    n_trials.append(self.n_trials)

                # save data
                all_res_continuous.append(res_continuous)
                all_res_earliest.append(res_earliest)
                all_session_nos.append(session_nos)
                all_session_names.append(session_names)
                all_n_trials.append(n_trials)
                all_sigs.append(sigs)

            if lockout==False:
                np.savez(self.main_dir+'/first_decoding_time'+
                         "_minTrials"+str(self.min_trials)+
                         '_all_'+
                         str(self.window)+'sec.npz',
                         all_res_continuous = all_res_continuous,
                         all_res_earliest = all_res_earliest,
                         all_session_nos = all_session_nos,
                         all_session_names = all_session_names,
                         all_n_trials = all_n_trials,
                         all_sigs = all_sigs
                         )
            else:
                np.savez(self.main_dir+'/first_decoding_time'+
                         "_minTrials"+str(self.min_trials)+
                         '_lockout_'+
                         str(self.window)+'sec.npz',
                         all_res_continuous = all_res_continuous,
                         all_res_earliest = all_res_earliest,
                         all_session_nos = all_session_nos,
                         all_session_names = all_session_names,
                         all_n_trials = all_n_trials,
                         all_sigs = all_sigs
                         )


    def compute_first_decoding_time_concatenated(self, lockouts=[False, True]):

        #
        #if lockouts = [False, True]
        for lockout in lockouts:
            self.lockout=lockout

            all_res_continuous = []
            all_res_earliest = []
            all_session_nos = []
            all_session_names = []
            all_n_trials = []
            all_sigs = []

            #
            for a in trange(len(self.animal_ids)):
                res_continuous = []
                res_earliest = []
                session_nos = []
                session_names = []
                n_trials = []
                sigs = []

                #
                self.animal_id = self.animal_ids[a]

                #
                self.get_sessions()
                #
                for p in range(len(self.session_ids)):
                    self.session_id = self.session_ids[p]

                    #
                    self.fname = self.fnames_svm[p]

                    self.process_session_concatenated()
                    # print ("a: ", a, self.session_id, self.data.shape)
                    #
                    if self.data.shape[0] == 0:
                        continue

                    # compute significance
                    self.compute_significance()

                    # save all the significant vals;
                    sigs.append(self.sig_save)

                    #
                    self.sig = self.sig.squeeze()

                    # find earliest period of significance, going back in time;
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            break

                    #
                    temp = -self.window+k/self.imaging_rate

                    # Exclude one of the weird datapoint from the AQ2? session
                    if temp>0:
                        #print ("n trials: ", self.n_trials, a,
                        #       p, temp, self.session_id, self.sig.shape)
                        continue

                    #
                    res_continuous.append(temp)

                    # find aboslute earliest
                    k_earliest = self.sig.shape[0]
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            k_earliest = k
                    res_earliest.append(-self.window+k_earliest/self.imaging_rate)

                    #
                    session_nos.append(p)

                    #
                    session_names.append(self.session_id)

                    #
                    n_trials.append(self.n_trials)

                # save data
                all_res_continuous.append(res_continuous)
                all_res_earliest.append(res_earliest)
                all_session_nos.append(session_nos)
                all_session_names.append(session_names)
                all_n_trials.append(n_trials)
                all_sigs.append(sigs)

            if lockout==False:
                np.savez(self.main_dir+'/first_decoding_time'+
                         "_concatenated.npz",
                         all_res_continuous = all_res_continuous,
                         all_res_earliest = all_res_earliest,
                         all_session_nos = all_session_nos,
                         all_session_names = all_session_names,
                         all_n_trials = all_n_trials,
                         all_sigs = all_sigs
                         )
            else:
                np.savez(self.main_dir+'/first_decoding_time'+
                         "_minTrials"+str(self.min_trials)+
                         '_lockout_'+
                         str(self.window)+'sec.npz',
                         all_res_continuous = all_res_continuous,
                         all_res_earliest = all_res_earliest,
                         all_session_nos = all_session_nos,
                         all_session_names = all_session_names,
                         all_n_trials = all_n_trials,
                         all_sigs = all_sigs
                         )

    def plot_first_decoding_time(self,
                                 return_ids_threshold,
                                 clrs):

        # flag to search for any signfiicant decoding time, not just continous ones
        earliest = False

        if earliest==False:

            data = np.load(self.main_dir+'/first_decoding_time'+
                         "_minTrials"+str(self.min_trials)+
                         '_all_'+
                         str(self.window)+'sec.npz',
                         allow_pickle=True)
            all_res_continuous_all = data['all_res_continuous']
            all_session_names = data['all_session_names']
            all_session_nos = data['all_session_nos']

            try:
                data = np.load(self.main_dir+'/first_decoding_time'+
                             "_minTrials"+str(self.min_trials)+
                             '_lockout_'+
                             str(self.window)+'sec.npz',
                             allow_pickle=True)
                all_res_continuous_lockout = data['all_res_continuous']
            except:
                all_res_continuous_lockout = []
                pass
            #all_n_trials = data['all_n_trials']
        else:
            print ("Data missing, skip")
            return

        if return_ids_threshold is not None:
            for k in range(len(all_res_continuous_all)):
                idx = np.where(np.array(all_res_continuous_all[k])<=return_ids_threshold)[0]
                if idx.shape[0]>0:
                    print ("all: ", np.array(all_session_names[k])[idx])
                    print ("all: ", np.array(all_session_nos[k])[idx])

            for k in range(len(all_res_continuous_lockout)):
                idx = np.where(np.array(all_res_continuous_lockout[k])<=return_ids_threshold)[0]
                if idx.shape[0]>0:
                    print ("lockout: ", np.array(all_session_names[k])[idx])
                    print ("lockout: ", np.array(all_session_nos[k])[idx])

        #
        data_sets_all = []
        for k in range(len(all_res_continuous_all)):
            data_sets_all.append(all_res_continuous_all[k])
        print (data_sets_all)
        #
        data_sets_lockout = []
        for k in range(len(all_res_continuous_lockout)):
            data_sets_lockout.append(all_res_continuous_lockout[k])

        # Computed quantities to aid plotting
        hist_range = (-self.window,1)
        bins = np.arange(-self.window,1,1)

        #
        binned_data_sets_all = [
            np.histogram(d, range=hist_range, bins=bins)[0]
            for d in data_sets_all
        ]

        binned_data_sets_lockout = [
            np.histogram(d, range=hist_range, bins=bins)[0]
            for d in data_sets_lockout
        ]

        #
        binned_maximums = np.max(binned_data_sets_all, axis=1)
        spacing = 40
        x_locations = np.arange(0, spacing*6,spacing)

        # The bin_edges are the same for all of the histograms
        bin_edges = np.arange(hist_range[0], hist_range[1],1)
        centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[1:]#[:-1]
        heights = np.diff(bin_edges)

        # Cycle through and plot each histogram
        fig, ax = plt.subplots(figsize=(10,5))
        for x_loc, binned_data, binned_data_lockout in zip(x_locations, binned_data_sets_all, binned_data_sets_lockout):
            lefts = x_loc - 0.3# * binned_data
            ax.barh(centers, -binned_data, height=heights, left=lefts, color=clrs[0])

            lefts = x_loc #- 0.5 * binned_data_lockout
            ax.barh(centers, binned_data_lockout, height=heights, left=lefts, color=clrs[1])

        ax.set_xticks(x_locations)
        ax.set_xticklabels(self.labels)
        ax.set_xlim(-20,spacing*6)
        ax.set_ylim(-self.window-0.5,0)
        ax.set_ylabel("Data values")
        ax.set_xlabel("Data sets")

    def exp_func(self, x, a, b, c, d):
        return a*np.exp(-c*(x-b))+d

    def fit_exp(self, all_preds,
                all_trials,
                ax):

        # fit exponentials
        all_preds = np.array(all_preds)
        offset = 10
        popt, pcov = curve_fit(self.exp_func,
                               all_preds+offset,
                               all_trials,
                               [offset,1e-6,0.001,0])

        #print("Popt: ", popt)
        time_back = -20
        x= np.linspace(time_back,offset,1000)
        ax.plot(x+time_back+offset,self.exp_func(x,*popt),
                 linewidth=5, c='black')

    def fit_line(self, all_preds,
                all_trials,
                color,
                ax):

        from sklearn import datasets, linear_model

        # Create linear regression object
        regr = linear_model.LinearRegression()

        #
        print (self.animal_id, self.session, all_preds)

        #
        all_preds = np.array(all_preds)[:,None]
        print ("all preds: ", all_preds.shape)
        all_trials = np.array(all_trials)
        regr.fit(all_preds, all_trials)

        # Make predictions using the testing set
        x_test = np.arange(-self.window,0.5,1)[:,None]
        y_test = regr.predict(x_test)

        #
        ax.plot(x_test, y_test,
                 color=color,
                 linewidth=3)


    def plot_first_decoding_time_vs_n_trials(self,
                                             clr,
                                             fname=None):
        #labels = ["M1", "M2", "M3", "M4","M5",'M6']

        # flag to search for any signfiicant decoding time, not just continous ones
        earliest = False

        if earliest==False:
            # data = np.load(self.main_dir + '/first_decoding_time_all_'+str(self.window)+
            #                'sec.npz',allow_pickle=True)
            if fname is None:
                fname = os.path.join(self.main_dir,
                                     '/first_decoding_time'+ "_minTrials"+str(self.min_trials)+
                                     '_all_'+
                                     str(self.window)+'sec.npz')

            data = np.load(fname,
                          allow_pickle=True)


            res_continuous_all = data['all_res_continuous']
            all_n_trials = data['all_n_trials']

            if self.lockout==True:
                try:
                    data = np.load(self.main_dir+'/first_decoding_time'+
                                 "_minTrials"+str(self.min_trials)+
                                 '_lockout_'+
                                 str(self.window)+'sec.npz',
                                 allow_pickle=True)
                    res_continuous_lockout = data['all_res_continuous']
                    lockout_n_trials = data['all_n_trials']
                except:
                    res_continuous_lockout=[]
                    lockout_n_trials = []
            else:
                res_continuous_lockout=[]
                lockout_n_trials = []
        else:
            print ("Data not found, skipping")
            return

        #
        fig = plt.figure(figsize=(20,20))
        all_preds = []
        all_trials = []
        import matplotlib.patches as mpatches

        for k in range(len(res_continuous_all)):
            ax=plt.subplot(2,3,k+1)
            plt.ylim(200,400)

            plt.xlim(self.xlim,0)
            plt.xticks([])
            #plt.yticks([])

            trials1 = np.array(all_n_trials[k])
            predictions1 = np.array(res_continuous_all[k])
            if predictions1.shape[0]<=1:
                continue

            #
            #print ("Predictiosn1: ", perdictions1.shape)
            plt.scatter(predictions1,
                        trials1,
                        s=100,
                        c=np.arange(trials1.shape[0])+20,
                        edgecolors='black',
                        cmap=cm.Reds)

            #
            #all_preds.extend(self.predictions1)
            #all_trials.extend(self.trials1)

            #self.fit_exp(all_preds, all_trials, ax)
            self.fit_line(predictions1,
                          trials1,
                          'black',
                          ax)

            #
            corr_pred_trials = scipy.stats.pearsonr(predictions1, trials1)
            corr_pred_time = scipy.stats.pearsonr(predictions1,
                                                  np.arange(len(predictions1)))


            # LOCKOUT TRIALS
            try:
                trials2 = np.array(lockout_n_trials[k])
                print ("lockout trials2: ", trials2)
                predictions2 = np.array(res_continuous_lockout[k])
                corr_pred_trial_lockout = scipy.stats.pearsonr(predictions2, trials2)
                corr_pred_time_lockout = scipy.stats.pearsonr(predictions2, np.arange(len(predictions2)))
            except:
                pass

            from decimal import Decimal
            patches = []
            patches.append(mpatches.Patch(color='black',
                                       label='all vs. # trials: '+
                                       str(round(corr_pred_trials[0],2))+
                                       " ("+str("pval: {0:.1}".format(corr_pred_trials[1]))+")"
                                          ))
            patches.append(mpatches.Patch(color='grey',
                                       label='all vs. time: '+
                                       str(round(corr_pred_time[0],2))+
                                       " ("+str("pval: {0:.1}".format(corr_pred_time[1]))+")"
                                        ))

            try:
                patches.append(mpatches.Patch(color='blue',
                                           label='lockout vs. # trials: '+
                                           str(round(corr_pred_trial_lockout[0],2))+
                                           " ("+str("pval: {0:.1}".format(corr_pred_trial_lockout[1]))+")"
                                          ))

                patches.append(mpatches.Patch(color='lightblue',
                                           label='lockout vs. time: '+
                                           str(round(corr_pred_time_lockout[0],2))+
                                           " ("+str("pval: {0:.1}".format(corr_pred_time_lockout[1]))+")"
                                          ))
            except:
                pass

            if True:
                plt.legend(handles=patches,
                           fontsize=6)

            try:
                plt.scatter(predictions2,
                            trials2,
                            s=100,
                            c=np.arange(trials2.shape[0])+20,
                            edgecolors='black',
                            cmap=cm.Blues)

                self.fit_line(predictions2,
                              trials2,
                              'blue',
                              ax)
            except:
                pass

            # select n_trials > 100 and connect them
            idx = np.where(trials1>80)[0]
            for p in idx:
                try:
                    plt.plot([predictions1[p], predictions2[p]],
                         [trials1[p], trials2[p]],'r--')
                except:
                    pass



        plt.suptitle("All sessions all trials")



    def plot_significant(self, clr, label):

        # set continuos to
        self.earliest_continuous = np.nan

        # GET FILENAME IF EXISTS
        self.get_fname()
        if self.fname is None:
            print ("no file, exiting")
            return

        # PROCESS SESSION
        self.process_session()
        self.n_trials_plotting.append(self.n_trials)
        if self.n_trials==0 or self.data.shape[0]==0:
            return
        print ("self n trials: ", self.n_trials)

        # COMPUTE TIME WINDOW FOR PLOTTING
        t = np.linspace(self.xlim[0], self.xlim[1], self.mean.shape[0])
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label + " # trials: "+str(self.n_trials),
                 linewidth=self.linewidth,
                 alpha=self.alpha)
        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # COMPUTE SIGNIFICANCE
        self.compute_significance()

        if self.show_EDT:
            self.ax.annotate("EDT: "+str(round(self.earliest_continuous,2))+"sec",
                         xy=(self.earliest_continuous, 0.5),
                         xytext=(self.earliest_continuous-3, 0.75+self.edt_offset),
                         arrowprops=dict(arrowstyle="->"),
                         fontsize=20,
                         color=clr)
            self.edt_offset+=0.02
            x = self.earliest_continuous

            #
            if True:
                plt.fill_between([x,0], 0,1.0 ,
                             color='grey',alpha=.2)

        # PLOT SIGNIFICANCE IMAGE BARS
        vmin=0.0
        vmax=self.significance
        axins = self.ax.inset_axes((0,1-self.cbar_thick-self.cbar_offset,1,self.cbar_thick))
        axins.set_xticks([])
        axins.set_yticks([])

        im = axins.imshow(self.sig,
                          vmin=vmin,
                          vmax=vmax,
                          aspect='auto',
                          #cmap='viridis_r')
                          cmap=self.cmap)

        #
        ticks = np.round(np.linspace(vmin, vmax, 4),8)
        print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
        #fmt = '%1.4f'
        fmt='%.0e'
        #
        if self.cbar:
            cbar = self.fig.colorbar(im,
                                ax=self.ax,
                                shrink=0.2,
                                ticks=ticks,
                                format = fmt)

            cbar.ax.tick_params(labelsize=25)

        # APPLY STANDARD FORMATS
        self.format_plot(self.ax)

        #
        try:
            fname = os.path.join(self.main_dir, self.animal_id,
                     'tif_files',
                     self.session,
                     'shift.txt'
                     )

            print (fname)
            shift = float(np.loadtxt(fname))

        except:
            shift = 0

        if self.show_title:
            plt.title(self.animal_id + "  session: "+str(self.session) +
                  "\n smoothing window: "+str(round(self.smooth_window/30.,1))+"sec"+
                  "\n [Ca] <-> DLC shift: "+str(round(shift,2))+" sec")

        #
        self.cbar_offset+=self.cbar_thick
        if self.show_legend:
            plt.legend(loc=self.legend_location,
                   fontsize=20)

        # for multi axes plots
        self.plotted = True

    #
    def plot_significant_concatenated(self, clr, label):

        #

        #
        data = np.load(self.fname, allow_pickle=True)
        self.data=data['accuracy']
        print ("loaded data: ", self.data.shape)

        #
        if self.smooth_window is not None:
            #mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

        #
        self.mean = self.data.mean(1)
        self.std = np.std(self.data, axis=1)

        #
        t = np.linspace(-self.window+2, 0,
                         self.mean.shape[0])

        #
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label,
                 linewidth=self.linewidth,
                 alpha=self.alpha)

        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # compute significance
        self.compute_significance()

        # set
        vmin=0.0
        vmax=self.significance

        # plot significance
        axins = self.ax.inset_axes((0,1-self.cbar_thick-self.cbar_offset,1,self.cbar_thick))
        axins.set_xticks([])
        axins.set_yticks([])
        im = axins.imshow(self.sig,
                          vmin=vmin,
                          vmax=vmax,
                          aspect='auto',
                          #cmap='viridis_r')
                          cmap=self.cmap)

        # find earliest significant;
        earliest_continuous = 0
        #print ("self sig: ", self.sig.shape)
        for k in range(self.sig.shape[1]-1,0,-1):
            if self.sig[0][k]<=self.significance:
                earliest_continuous = k
            else:
                break

        print ("earliest: ", earliest_continuous,
               " in sec: ", -(self.sig.shape[1]-earliest_continuous)/30.)


        self.ax.annotate("EDT: "+str(round(-(self.sig.shape[1]-earliest_continuous)/30.,1))+"sec",
                         xy=(-(self.sig.shape[1]-earliest_continuous)/30., 0.5),
                         xytext=(-(self.sig.shape[1]-earliest_continuous)/30.-12, 0.75+
                                 self.title_offset),
                         arrowprops=dict(arrowstyle="->",
                                         lw=5),
                         fontsize=20,
                         color=clr)
        x = -(self.sig.shape[1]-earliest_continuous)/30.

        # plot significance region
        plt.fill_between([x-1/33.,0], 0,1.0 ,
                         color=clr,alpha=.2)


        if self.cbar==True:

            #
            ticks = np.round(np.linspace(vmin, vmax, 4),8)
            print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
            #fmt = '%1.4f'
            fmt='%.0e'


            #
            cbar = self.fig.colorbar(im,
                                ax=self.ax,
                                shrink=0.2,
                                ticks=ticks,
                                format = fmt)

            cbar.ax.tick_params(labelsize=25)

            self.format_plot(self.ax)
            plt.title(self.animal_id + "  session: "+str(self.session))

            #
            self.cbar_offset+=self.cbar_thick


        #

    def plot_trends(self, clr, label, ax):
        #
        t = np.arange(-9.5, 10.5, 1)

        colors = plt.cm.magma(np.linspace(0,1,self.n_colors))

        #
        mean = self.data.mean(1)
        std = np.std(self.data,axis=1)
        ax.plot(t, mean,
                 c=colors[clr],
                 label = label,
                 linewidth=4)

        self.format_plot(ax)


    def plot_animal_decision_longitudinal_matrix(self,
                                                 animal_name,
                                                 lockout=False,
                                                 ax=None):


        if ax is None:
            fig=plt.figure()
            ax=plt.subplot(111)
        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        #fig=plt.figure()
        img =[]
        for fname in fnames:
            if 'lockout' not in fname:
                if lockout==False:
                    self.load_data(fname)
                    temp = self.data.mean(1)
                else:
                    #
                    idx = fname.find('SVM_scores_'+animal_name)
                    fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                    self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                    temp = self.data.mean(1)

                if self.filter:
                    temp = self.filter(temp)
                img.append(temp)

        img=np.array(img)
        ax.imshow(img)

        plt.xticks(np.arange(0,img.shape[1],2),
                           np.arange(-9.5,10,2))
        plt.ylabel("Study day")
        plt.title(animal_name)
        plt.suptitle("lockout: "+str(lockout))
        #ticks = np.linspace(vmin,vmax,4)
        #cbar = fig.colorbar(im, ax=axes[5],
        #                    ticks=ticks)


    def plot_animal_decision_longitudinal(self, animal_name):
        #animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=self.animal_ids.index(animal_name)

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.set_title("All")
        ax2=plt.subplot(1,2,2)
        ax2.set_title("Lockout")

        self.n_colors = self.n_colors[idx]
        ctr=0
        for fname in fnames:
            if 'lockout' not in fname:
                print (fname)
                self.load_data(fname)
                self.plot_trends(ctr,'all',ax1)

                idx = fname.find('SVM_scores_'+animal_name)
                fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                print (fname2)
                self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                self.plot_trends(ctr,'lockout', ax2)
                ctr+=1
        plt.suptitle(animal_name)

    def get_sessions(self):

        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        # load sessions in chronological order
        self.sessions = np.load(os.path.join(self.main_dir, self.animal_id,'tif_files.npy'))

        #
        self.fnames_svm = []
        self.session_ids = []
        for k in range(len(self.sessions)):
            self.session = os.path.split(self.sessions[k])[1][:-4]
            self.session_ids.append(self.session)

            fname = os.path.join(self.main_dir, self.animal_id,
                     'tif_files',
                     self.session,
                     self.session+
                     '_globalPca_min_trials_concatenated200_code_04_30sec_accuracy.npz'
                     )

            self.fnames_svm.append(fname)


    def plot_animal_time_longitudinal(self, animal_name):
        #animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=self.animal_ids.index(animal_name)

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

        #
        fig=plt.figure()
        ax1=plt.subplot(1,2,1)
        ax1.set_title("All")
        ax2=plt.subplot(1,2,2)
        ax2.set_title("Lockout")

        self.n_colors = self.ncolors[idx]
        ctr=0
        for fname in fnames:
            if 'lockout' not in fname:
                print (fname)
                self.load_data(fname)
                self.plot_trends(ctr,'all',ax1)

                idx = fname.find('SVM_scores_'+animal_name)
                fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                print (fname2)
                self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                self.plot_trends(ctr,'lockout', ax2)
                ctr+=1

        plt.suptitle(animal_name)



    def plot_animal_decision_AUC_longitudinal(self):

            #
        ax1=plt.subplot(121)
        ax2=plt.subplot(122)
        #
        for animal_name in self.animal_ids:
            #
            root_dir = self.main_dir+animal_name+'/'
            fnames = np.sort(glob2.glob(root_dir+'SVM_scores_'+animal_name+"*"))

            #
            width = [0,20]

            #
            auc1 = []
            auc2 = []
            for fname in fnames:
                if 'lockout' not in fname:
                    self.load_data(fname)
                    auc1.append(self.data.mean(1)[width[0]:width[1]].sum(0))

                    # load lockout version
                    idx = fname.find('SVM_scores_'+animal_name)
                    fname2 = fname[:idx+12]+"_lockout"+fname[idx+12:]
                    self.load_data(fname[:idx+14]+"_lockout"+fname[idx+14:])
                    auc2.append(self.data.mean(1)[width[0]:width[1]].sum(0))

            #
            auc1 = np.array(auc1)
            auc2 = np.array(auc2)
            t = np.arange(auc1.shape[0])/(auc1.shape[0]-1)

            #
            print ("anmial: ", animal_name, t.shape, auc1.shape)
            temp2 = np.poly1d(np.polyfit(t, auc1, 1))(t)
            ax1.plot(t, temp2,
                     linewidth=4,
                     c=self.colors[self.clr_ctr],
                    label=self.animal_ids[self.clr_ctr])

            #
            ax2.plot(t, np.poly1d(np.polyfit(t, auc2, 1))(t),
                     '--', linewidth=4,
                     c=self.colors[self.clr_ctr])

            self.clr_ctr+=1

        ax1.set_xlim(0,1)
        ax2.set_xlim(0,1)
        ax1.set_ylim(9,13)
        ax2.set_ylim(9,13)
        ax1.set_title("All")
        ax2.set_title("Lockout")
        ax1.set_xlabel("Duration of study")
        ax2.set_xlabel("Duration of study")
        plt.suptitle("AUC fits to SVM decision prediction", fontsize=20)
        ax1.legend(fontsize=20)


    def plot_decision_time(self, clr, label, ax=None):

        #
        if ax is None:
            ax=plt.subplot(111)

        #
        t = np.arange(-9.5, 0.5, 1)

        #
        print (self.data.shape)

        temp = []
        for k in range(self.data.shape[1]):
            temp.append(self.data[:,k,k])
        temp=np.array(temp)

        #
        mean = temp.mean(1)
        std = np.std(temp,axis=1)
        plt.plot(t, mean,
                 c=clr,
                 label = label,
                 linewidth=4)
        plt.fill_between(t, mean-std, mean+std, color=clr, alpha = 0.1)

        plt.legend(fontsize=16)
        self.format_plot2(ax)



    def plot_decision_time_animal(self, animal_name):

        # select dataset and # of recordings
        t = np.arange(-9.5, 0.5, 1)
        idx=self.animal_ids.index(animal_name)

        colors = plt.cm.magma(np.linspace(0,1,self.n_colors[idx]))


        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'conf_10_'+animal_name+"*"))

        #
        traces1 = []
        traces2 = []
        for fname in fnames:
            if 'lockout' not in fname:
                self.load_data(fname)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k])
                traces1.append(temp)

                # load lockout version
                idx = fname.find('conf_10_'+animal_name)
                fname2 = fname[:idx+11]+"_lockout"+fname[idx+11:]
                self.load_data(fname2)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k])
                traces2.append(temp)

        traces1=np.array(traces1)
        traces2=np.array(traces2)
        #print (traces1.shape)
        #
        ax1=plt.subplot(1,2,1)
        ax1.set_title("all")
        for k in range(traces1.shape[0]):
            mean=traces1[k].mean(1)
            #print (mean.shape)
            ax1.plot(t, mean,
                     c=colors[k],
                     linewidth=4)

        self.format_plot2(ax1)

        #
        ax2=plt.subplot(1,2,2)
        ax2.set_title("lockout")
        for k in range(traces2.shape[0]):
            mean=traces2[k].mean(1)
            #print (mean.shape)
            ax2.plot(t, mean,
                     c=colors[k],
                     linewidth=4)
        self.format_plot2(ax2)

        plt.suptitle(animal_name)


    #
    # def save_edts_body_movements(self, animal_ids, codes):
    #
    #     #
    #     for animal_id in animal_ids:
    #         self.animal_id= animal_id
    #
    #         #
    #         fnames_good = self.main_dir+self.animal_id + '/tif_files/sessions_DLC_alignment_good.txt'
    #
    #         import csv
    #         sessions = []
    #         shift_ids = []
    #         with open(fnames_good, newline='') as csvfile:
    #             spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #             for row in spamreader:
    #                 sessions.append(str(row[0]).replace(',',''))
    #                 shift_ids.append(row[1])
    #
    #         print ("SESSIONS: ", sessions)
    #         print ("SHIFT IDS: ", shift_ids)
    #
    #         ############################
    #
    #         #
    #         edt = []
    #         n_trials = []
    #
    #         # loop over all sessions
    #         for ctr,session_id in enumerate(sessions):
    #             edt.append([])
    #             n_trials.append([])
    #             # get the shifted data by matching session ID with all sessions
    #             self.session_id = session_id
    #             for k in range(len(sessions)):
    #                 if self.session_id in sessions[k]:
    #                     ctr_plt = k
    #                     break
    #
    #             # loop over all limb movements
    #             self.xlim = [-15, 0]
    #             for code in codes:
    #
    #                 self.code = code
    #                 self.shift_id_str = shift_ids[ctr_plt]
    #
    #                 # GET FILENAME IF EXISTS
    #                 self.get_fname()
    #                 if self.fname is None:
    #                     print ("no file, exiting")
    #                     edt[ctr].append(np.nan)
    #                     n_trials[ctr].append(0)
    #                     continue
    #
    #                 # PROCESS SESSION
    #                 self.process_session()
    #                 if self.n_trials==0 or self.data.shape[0]==0:
    #                     #print ("insufficient trials ", labels[i])
    #                     edt[ctr].append(np.nan)
    #                     n_trials[ctr].append(0)
    #                     continue
    #
    #                 # GET SIGNIFICANCE AND EDT
    #                 self.compute_significance()
    #                 edt[ctr].append(self.edt)
    #                 n_trials[ctr].append(self.n_trials)
    #                 print ("*******************************")
    #                 print ("    EDT: ", code, " ", self.edt)
    #                 print ("*******************************")
    #                 print ('')
    #                 print ('')
    #
    #         n_trials = np.array(n_trials)
    #         edt = np.array(edt)
    #         print (edt.shape)
    #
    #         # data_dir = '/media/cat/4TBSSD/yuki/'
    #         np.savez(self.main_dir + animal_id+"_edt_body_parts.npz",
    #                 edt = edt,
    #                 n_trials = n_trials)


    def plot_decision_time_animal_matrix(self, animal_name):

        #
        root_dir = self.main_dir+animal_name+'/'
        fnames = np.sort(glob2.glob(root_dir+'conf_10_'+animal_name+"*"))

        #
        traces1 = []
        traces2 = []
        for fname in fnames:
            if 'lockout' not in fname:
                self.load_data(fname)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k].mean(0))
                traces1.append(temp)

                # load lockout version
                idx = fname.find('conf_10_'+animal_name)
                fname2 = fname[:idx+11]+"_lockout"+fname[idx+11:]
                self.load_data(fname2)
                temp = []
                for k in range(self.data.shape[1]):
                    temp.append(self.data[:,k,k].mean(0))
                traces2.append(temp)

        traces1=np.array(traces1)
        traces2=np.array(traces2)
        print (traces1.shape)
        #
        ax1=plt.subplot(1,4,1)
        ax1.set_title("all")
        ax1.imshow(traces1,vmin=0,vmax=1.0)

        #
        ax2=plt.subplot(1,4,2)
        ax2.set_title("lockout")
        ax2.imshow(traces2,vmin=0,vmax=1.0)
        plt.suptitle(animal_name)

        #
        ax2=plt.subplot(1,4,3)
        ax2.set_title("all - lockout")
        ax2.imshow(traces1-traces2,vmin=0,vmax=.25)
        plt.suptitle(animal_name)

        #
        ax2=plt.subplot(1,4,4)
        ax2.set_title("lockout - all")
        ax2.imshow(traces2-traces1,vmin=0,vmax=.25)
        plt.suptitle(animal_name)

    def plot_decision_time_all_matrix(self):


        vmin = 0
        vmax = 0.75
        axes=[]
        fig=plt.figure()
        for a in range(6):
            axes.append(plt.subplot(2,3,a+1))
            #
            root_dir = self.main_dir+self.animal_ids[a]+'/'
            fnames = np.sort(glob2.glob(root_dir+'conf_10_'+self.animal_ids[a]+"*"))

            #
            traces1 = []
            for fname in fnames:
                if 'lockout' not in fname:
                    self.load_data(fname)
                    temp = []
                    for k in range(self.data.shape[1]):
                        temp.append(self.data[:,k,k].mean(0))
                    if self.filter:
                        temp = self.filter_trace(temp)
                    traces1.append(temp)

            traces1=np.array(traces1)
            axes[a].set_title(self.animal_ids[a])
            im = axes[a].imshow(traces1,vmin=vmin,vmax=vmax)

            plt.xticks(np.arange(0,traces1.shape[1],2),
                               np.arange(-9.5,0,2))
            plt.ylabel("Study day")

        ticks = np.linspace(vmin,vmax,4)
        cbar = fig.colorbar(im, ax=axes[5],
                            ticks=ticks)
        #cbar.ax.tick_params(labelsize=16)
        #cbar.ax.set_title('Pval', rotation=0,
        #                 fontsize=16)


    def filter_trace(self,trace):

        box = np.ones(self.smooth_window)/self.smooth_window
        trace_smooth = np.convolve(trace, box, mode='valid')

        return trace_smooth


    def compare_svm_rnn(self, fnames_svm, fnames_rnn):

        #
        ctr=1
        for fname in fnames_rnn:
            idx1 = fname.find('RNN_scores/')
            idx2 = fname.find('_200')
            session = fname[idx1+11:idx2]
            for fname_svm in fnames_svm:
                if session in fname_svm and 'trial' in fname_svm:
                    data_svm = np.load(fname_svm)[:300]
                    break

            #
            ax=plt.subplot(5, 10,ctr)
            data_rnn = np.load(fname)['b_rnn']
            t = np.linspace(-10,0,data_rnn.shape[0])
            std = np.std(data_rnn,1)
            mean = data_rnn.mean(1)
            plt.plot(t, mean, c='black')
            plt.fill_between(t, mean+std, mean-std, color='black', alpha=.2)


            mean = data_svm.mean(1)
            self.smooth_window = 30
            mean = self.filter_trace(mean)
            std = np.std(data_svm,1)[15:-14]

            t = np.linspace(-10,0,mean.shape[0])
            plt.plot(t,mean,c='blue')
            plt.fill_between(t, mean+std, mean-std, color='blue', alpha=.2)

            plt.ylim(0.4,1.0)
            plt.title(os.path.split(fname_svm)[1][11:25],fontsize=8)
            plt.plot([-10,0],[0.5,0.5],'r--')
            plt.xlim(-10,0)
            if ctr!=31:
                plt.yticks([])
                plt.xticks([])

            ctr+=1

        plt.suptitle("Decision choice: RNN (black) vs. SVM (blue) ",fontsize=20)
        plt.show()

def get_lever_offset(main_dir,
                     animal_id,
                     session_corrected):

    fname_lever_offset = os.path.join(main_dir,
                                     animal_id,
                                     'tif_files',
                                     session_corrected,
                                     session_corrected+"_lever_offset_n_frames.txt")

    if os.path.exists(fname_lever_offset)==False:

        images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

        aligned_images = np.load(images_file)

        # Find blue light on/off
        blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
        start_blue = 0; end_blue = aligned_images.shape[0]

        if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])< blue_light_threshold:
                    #self.aligned_images = self.aligned_images[k:]
                    end_blue = k
                    break
        else:                                                           #Case #2: start with light off; remove starting and end chunks;
            #Find first light on
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])> blue_light_threshold:
                    start_blue = k
                    break

            #Find light off - count backwards from end of imaging data
            for k in range(len(aligned_images)-1,0,-1):
                if np.average(aligned_images[k])> blue_light_threshold:
                    end_blue= k
                    break

        lever_offset = start_blue

        #np.savetxt(fname_lever_offset, [self.lever_offset])

    else:
        lever_offset = int(np.loadtxt(fname_lever_offset))

    return lever_offset





def get_sessions(main_dir,
                 animal_id,
                 session_id):
     # load ordered sessions from file
    sessions = np.load(os.path.join(main_dir,
                                         animal_id,
                                         'tif_files.npy'))
    # grab session names from saved .npy files
    data = []
    for k in range(len(sessions)):
        data.append(os.path.split(sessions[k])[1].replace('.tif',''))
    sessions = data

    #
    if session_id != 'all':
        final_session = []
        for k in range(len(sessions)):
            if session_id in sessions[k]:
                final_session = [sessions[k]]
                break
        sessions = final_session

    # fix binary string files issues; remove 'b and ' from file names
    for k in range(len(sessions)):
        sessions[k] = str(sessions[k]).replace("'b",'').replace("'","")
        if sessions[k][0]=='b':
            sessions[k] = sessions[k][1:]

    sessions = np.array(sessions)

    return sessions


def load_trial_times_whole_stack(root_dir,
                                 animal_id,
                                 session,
                                 no_movement):

    # grab movement initiation arrays
    fname = os.path.join(root_dir, animal_id,'tif_files',
                         session,
                         session+'_'+
                         str(no_movement)+"secNoMove_movements.npz"
                         )

    # if no file return empty arrays?
    if os.path.exists(fname)==False:
        feature_quiescent = []
        #
        for k in range(7):
            feature_quiescent.append([])

        return None, None, None
    #
    data = np.load(fname, allow_pickle=True)
    feature_quiescent = data['feature_quiescent']
    all_quiescent = data['all_quiescent']

    # load rewarded lever pull trigger times also
    code_04_times, code_04_times_lockout = load_code04_times(root_dir,
                                                              animal_id,
                                                              no_movement,
                                                              session)
    code_04_times = np.array((code_04_times, code_04_times)).T
    shift_lever_to_ca = get_lever_offset_seconds(root_dir,
                                                 animal_id,
                                                 session
                                                 )
    print ("Lever to [Ca] shift: ", shift_lever_to_ca)

    #
    bins = np.arange(-10,10,1/15.)

    #
    try:
        res = pycorrelate.pcorrelate(code_04_times[:,1],
                                 np.array(feature_quiescent[1])[:,1],
                                 bins=bins)
    except:
		
        try:
            res = pycorrelate.pcorrelate(code_04_times[:,1],
                         np.array(feature_quiescent[0])[:,1],
                         bins=bins)
        except:
            return None, None, None


    argmax = np.argmax(res)
    shift_DLC_to_ca = bins[argmax]

    #
    # shift_DLC_to_ca = get_DLC_shift_seconds(root_dir,
    #                                         animal_id,
    #                                         session,
    #                                         session_number)

    print ("DLC to [Ca] shift: ", shift_DLC_to_ca)

    #code_04_times += shift_lever_to_ca

    #load_code04_times = code_04_times
    #feature_quiescent = feature_quiescent

    #
    temp_ = []
    for k in range(len(feature_quiescent)):
        temp_.append(np.array(feature_quiescent[k])-shift_lever_to_ca)
    temp_.append(all_quiescent)
    temp_.append(code_04_times - shift_DLC_to_ca - shift_lever_to_ca)

    return temp_, code_04_times, feature_quiescent



def get_DLC_shift_seconds(main_dir,
                          animal_id,
                          session,
                          session_number):

    fnames_good = os.path.join(main_dir,animal_id,'tif_files',
                  'sessions_DLC_alignment_good.txt')

    import csv
    sessions = []
    shift_ids = []
    with open(fnames_good, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            sessions.append(str(row[0]))
            shift_ids.append(row[1])

    shift_id_str = shift_ids[session_number]

    fname_correlate = os.path.join(main_dir, animal_id,
                         'tif_files', session,
                        'correlate.npz')

    try:
        data = np.load(fname_correlate,allow_pickle=True)
    except:
        print( " ... data missing", fname_correlate)
        return None

    cors = data['cors'].squeeze().T

    #vis.shift = 0
    print ("sessoin ID: ", session_number, "  left/right paw/lever ID used: ", shift_id_str)
    if len(shift_id_str)>1:
        shift_id = int(shift_id_str[0])
        shift_additional = float(shift_id_str[1:])
    else:
        shift_id = int(shift_id_str)
        shift_additional = 0

    print ( " using shift: ", shift_id+shift_additional)

    corr_featur_id = shift_id

    temp_trace = cors[:,corr_featur_id]
    temp_trace[:2000] = 0
    temp_trace[-2000:] = 0
    shift = round(np.argmax(temp_trace)/1000. - 15.,2)+shift_additional
    print ("SHIFT Loaded: ", shift)

    return shift

def load_code04_times(root_dir,
                      animal_id,
                      lockout_window,
                      recording):

    #
    try:
        fname = os.path.join(root_dir,animal_id, 'tif_files',recording,
                             recording + '_locs44threshold.npy')
        locs_44threshold = np.load(fname)
    except:
        print ("locs 44 thrshold missing", recording)
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    #
    codes = np.load(os.path.join(root_dir,animal_id, 'tif_files',recording,
                             recording + '_code44threshold.npy'))
    code = b'04'
    idx = np.where(codes==code)[0]
    locs_selected = locs_44threshold[idx]

    if locs_selected.shape[0]==0:
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    diffs = locs_selected[1:]-locs_selected[:-1]
    idx = np.where(diffs>lockout_window)[0]

    #
    locs_selected_with_lockout = locs_selected[idx+1]
    if locs_selected_with_lockout.shape[0]==0:
        locs_code04 = np.zeros((0),'float32')
        locs_code04_lockout = np.zeros((0),'float32')
        return locs_code04, locs_code04_lockout

    # ADD FIRST VAL
    if locs_selected[0]>lockout_window:
        locs_selected_with_lockout = np.concatenate(([locs_selected[0]], locs_selected_with_lockout), axis=0)

    locs_code04 = locs_selected
    locs_code04_lockout = locs_selected_with_lockout

    return locs_code04, locs_code04_lockout


def get_lever_offset_seconds(main_dir,
                     animal_id,
                     session_corrected,
                     imaging_rate=30):

    fname_lever_offset = os.path.join(main_dir,
                                     animal_id,
                                     'tif_files',
                                     session_corrected,
                                     session_corrected+"_lever_offset_n_frames.txt")

    if os.path.exists(fname_lever_offset)==False:

        images_file = fname_lever_offset.replace('_lever_offset_n_frames.txt','_aligned.npy')

        aligned_images = np.load(images_file)

        # Find blue light on/off
        blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
        start_blue = 0; end_blue = aligned_images.shape[0]

        if np.average(aligned_images[0])> blue_light_threshold:    #Case #1: imaging starts with light on; need to remove end chunk; though likely bad recording
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])< blue_light_threshold:
                    #self.aligned_images = self.aligned_images[k:]
                    end_blue = k
                    break
        else:                                                           #Case #2: start with light off; remove starting and end chunks;
            #Find first light on
            for k in range(len(aligned_images)):
                if np.average(aligned_images[k])> blue_light_threshold:
                    start_blue = k
                    break

            #Find light off - count backwards from end of imaging data
            for k in range(len(aligned_images)-1,0,-1):
                if np.average(aligned_images[k])> blue_light_threshold:
                    end_blue= k
                    break

        lever_offset = start_blue

        #np.savetxt(fname_lever_offset, [self.lever_offset])

    else:
        lever_offset = int(np.loadtxt(fname_lever_offset))

    return lever_offset/imaging_rate


#
def plot_vertical_histograms(animal_ids,
                             code_ids,
                             clrs,
                             window,
                             return_ids_threshold):

    all_res1 = []
    all_res2 = []
    for animal_id in animal_ids:
        fname = '/media/cat/4TBSSD/yuki/'+animal_id+'_edt_body_parts.npz'

        data = np.load(fname, allow_pickle=True)
        #

        edt1 = data['edt'][:,code_ids[0]]
        all_res1.append(edt1)

        edt2 = data['edt'][:, code_ids[1]]
        all_res2.append(edt2)

        #
        if return_ids_threshold is not None:
            #for k in range(len(edt1)):
            idx = np.where(edt1<=return_ids_threshold)[0]
            if idx.shape[0]>0:
                print (animal_id, "session 1 with EDT < threshold: ", idx)

            idx = np.where(edt2<=return_ids_threshold)[0]
            if idx.shape[0]>0:
                print (animal_id, "session 2 with EDT < threshold: ", idx)
    #
    data_sets_all = []
    for k in range(len(all_res1)):
        data_sets_all.append(all_res1[k])

    #
    data_sets_lockout = []
    for k in range(len(all_res2)):
        data_sets_lockout.append(all_res2[k])

    # Computed quantities to aid plotting
    hist_range = (-window,1)
    bins = np.arange(-window,1,1)

    #
    binned_data_sets_all = [
        np.histogram(d, range=hist_range, bins=bins)[0]
        for d in data_sets_all
    ]

    binned_data_sets_lockout = [
        np.histogram(d, range=hist_range, bins=bins)[0]
        for d in data_sets_lockout
    ]

    #
    binned_maximums = np.max(binned_data_sets_all, axis=1)
    spacing = 30
    x_locations = np.arange(0, spacing*6,spacing)

    # The bin_edges are the same for all of the histograms
    bin_edges = np.arange(hist_range[0], hist_range[1],1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[1:]#[:-1]
    heights = np.diff(bin_edges)

    # Cycle through and plot each histogram
    fig, ax = plt.subplots(figsize=(6,5))
    for x_loc, binned_data, binned_data_lockout in zip(x_locations, binned_data_sets_all, binned_data_sets_lockout):
        lefts = x_loc - 0.3# * binned_data
        ax.barh(centers, -binned_data, height=heights, left=lefts, color=clrs[code_ids[0]])

        lefts = x_loc #- 0.5 * binned_data_lockout
        ax.barh(centers, binned_data_lockout, height=heights, left=lefts, color=clrs[code_ids[1]])

    ax.set_xticks(x_locations)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_xticklabels(labels)
    ax.set_xlim(-15,spacing*6-5)
    ax.set_ylim(-window-0.5,0)
    #ax.set_ylabel("Data values")
    #ax.set_xlabel("Data sets")


def plot_pca_timecourses(animal_id,
                         session_id,
                        class_type,
                        n_pca,
                        xlim,
                         plotting):

    class_types = ['trial','random']
    clrs = ['blue','black']
    ctr=0
    if plotting:
        fig =plt.figure()
    for class_type in class_types:
        #
        fname = '/media/cat/4TBSSD/yuki/'+animal_id+'/tif_files/'+session_id+"/"+session_id+'_code_04_trial_ROItimeCourses_30sec_pca_0.95.npy'
        fname = fname.replace('trial',class_type)
        print ("fname: ", fname)

        fname_spatial = '/media/cat/4TBSSD/yuki/'+animal_id+'/tif_files/'+session_id+"/"+session_id+'_code_04_trial_ROItimeCourses_30sec_pca_0.95_spatial.npy'
        pca_space =np.load(fname_spatial).reshape(-1, 128,128)
        print ("pca_space: ", pca_space.shape)

        #
        pca_trials = np.load(fname)
        pca_trials = pca_trials[:,:,:900]
        n_pca = min(pca_trials.shape[1],n_pca)
        pca_trials = pca_trials[:,:n_pca,:900]
        print ("pca_trials: ", pca_trials.shape)

        #
        means = np.mean(pca_trials,0)
        std = np.std(pca_trials,0)
        t = np.arange(means.shape[1])/30 - 30.

        if class_type=='trial':
            ylims = []

        if plotting:
            for k in range(means.shape[0]):
                ax=plt.subplot(3,n_pca,ctr+1)

                for p in range(pca_trials.shape[0]):
                    plt.plot(t,pca_trials[p,k],
                             c=clrs[ctr//n_pca],
                             alpha=.1)
                    if p>50:
                        break

                plt.plot(t,means[k],
                        linewidth=2,
                        c='red',
                        label="PC: "+str(k+1))
                plt.legend(fontsize=20)


                if class_type=='trial':
                    vmax = max(-np.min(means[k]),
                               np.max(means[k]))*5
                    plt.ylim(-vmax,vmax)
                    ylims.append(vmax)
                else:
                    plt.ylim(-ylims[k],ylims[k])


                if ctr==0:
                    plt.ylabel("Trials")
                elif ctr==n_pca:
                    plt.ylabel("Random")

                plt.xlim(xlim[0],0)
                ctr+=1
            plt.xlim(t[0],t[-1])


    if plotting:
        for k in range(n_pca):
            ax=plt.subplot(3,n_pca,ctr+1)
            if k==0:
                plt.ylabel("PCA loadings")
            plt.imshow(pca_space[k])
            ctr+=1

        plt.suptitle(session_id+ "  # trials: "+str(pca_trials.shape[0]), fontsize=20)
        plt.show()

    return pca_trials



class PCA_Analysis():

    def __init__(self):

        self.clr_ctr = 0


def get_data_and_triggers(pa):

    #
    fname_triggers = os.path.join(pa.root_dir,
                                  pa.animal_id,
                                  'tif_files',
                                  pa.session,
                                  'blue_light_frame_triggers.npz')

    fname_data = os.path.join(pa.root_dir,
                                 pa.animal_id,
                                 'tif_files',
                                 pa.session,
                                 pa.session+ '_aligned_butterworth_0.1hz_6.0hz.npy')

    try:
        meta = np.load(fname_triggers)
        data = np.load(fname_data)
    except:
        print (" data missing/led missing")
        return None, None

    end_blue = meta['end_blue']
    start_blue = meta['start_blue']
    triggers = meta['img_frame_triggers']

    #print (" triggers: ", triggers.shape)

    #
    #print (" data: ", data.shape)

    data_led = data[start_blue:end_blue]
    #print (data_led.shape)

    X = data_led.reshape(data_led.shape[0],-1)
    #print ("X: ", X.shape)

    return X, triggers


def get_pca_object_and_all_points(pa):

    fname_pickle = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_pca_scatter_plot.pk")

    fname_all_points = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_all_points.npy")
    X_30 = []
    for k in range(0,pa.X.shape[0]-100,pa.sliding_window):
        X_30.append(pa.X[k:k+pa.sliding_window])
    X_30 = np.array(X_30)
    X_30 = X_30.reshape(X_30.shape[0],-1)
    print(" X data using : ", pa.sliding_window, " number of frames ", X_30.shape)

    if os.path.exists(fname_all_points)==False:

        # PCA ON ALL DATA
        pca = PCA(n_components=pa.n_pca)
        pca.fit(X_30)
        print (" done fit ")

        # do
        all_points = pca.transform(X_30)
        print ("all points denoised: ", all_points.shape)

        with open(fname_pickle, 'wb') as f:
            pickle.dump(pca, f)

        np.save(fname_all_points,
                all_points)
    else:

        with open(fname_pickle, "rb") as f:
            pca = pickle.load(f)
        all_points = np.load(fname_all_points)

    return pca, all_points


def get_umap_object_and_all_points(pa):

    fname_pickle = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_umap_scatter_plot.pk")

    fname_all_points = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session,
                                pa.session+"_umap_all_points.npy")
    # use raw 128 x 128 data
    if False:
        X_30 = []
        for k in range(0,pa.X.shape[0]-100,pa.sliding_window):
            X_30.append(pa.X[k:k+pa.sliding_window])
        X_30 = np.array(X_30)
        X_30 = X_30.reshape(X_30.shape[0],-1)
    else:
        X_30 = pa.all_points

    print(" X data using : ", pa.sliding_window, " number of frames ", X_30.shape)

    if os.path.exists(fname_all_points)==False:

        # PCA ON ALL DATA

        import umap
        umap_3d = umap.UMAP(n_components=3,
                            init='random',
                            random_state=0)

        #fit = umap.UMAP()
        print ("Fitting UMAP...")
        umap_3d.fit(X_30)
        print ("  ... done fit ")

        # do
        print (" denoising all data using UMAP ....", X_30.shape)
        all_points = umap_3d.transform(X_30)
        #all_points = pca.transform(X_30)
        print ("all points denoised: ", all_points.shape)

        with open(fname_pickle, 'wb') as f:
            pickle.dump(umap_3d, f)

        np.save(fname_all_points,
                all_points)
    else:

        with open(fname_pickle, "rb") as f:
            umap_3d = pickle.load(f)
        all_points = np.load(fname_all_points)

    return umap_3d, all_points



def project_data_pca(pa):

    # p_lever = np.zeros((pa.n_frames,
    #                     pa.triggers.shape[0],
    #                     pa.n_pca))
    print (pa.n_frames,
           pa.triggers.shape[0],
           pa.n_pca)

    #
    # total_frames = pa.n_frames
    p_lever = []
    for t in trange(pa.n_frames):
        arr = []
        #if True:
        try:
            for k in range(pa.triggers.shape[0]):
                temp = pa.X[pa.triggers[k]-pa.sliding_window -t:pa.triggers[k]-t]
                if temp.shape[0]== pa.sliding_window:
                    temp = temp.reshape(temp.shape[0],
                                   -1)
                    #X_lever[t,k]= temp
                    arr.append(temp.reshape(-1))

            arr = np.array(arr)
            res = pa.pca.transform(arr)
            #p_lever[t] = res
            p_lever.append(res)

        except:
            pass
    p_lever = np.array(p_lever)
    #
    # # remove any skipped values
    # if total_frames<pa.n_frames:
    #     p_lever = p_lever[:total_frames]

    #print (" pca p_levefr resahped: ", p_lever.shape)

    return p_lever



def project_data_umap(pa):

    # DIMENSION OF DATA IN:
    # pa.p_lever = (1, 55, 10)

    #
    p_lever = []
    for t in trange(pa.n_frames):
        #for k in range(pa.p_lever.shape[1]):
        #    temp = pa.p_lever[t,k]

        arr = pa.p_lever[t]
        print ("Umap applid to arr: ", arr.shape)
        res = pa.umap.transform(arr)
        #p_lever[t] = res
        p_lever.append(res)

    p_lever = np.array(p_lever)

    print (" umap p_levefr resahped: ", p_lever.shape)

    return p_lever
def plot_pca_scatter_multi(pa,
                     n_frames = 30,
                     clr = 'red',
                     plot_all = True,
                     plot_3D=True,
                     plot_flag = True):

    # use knn triage to remove most outlier points
    triage_value = 0.0008
    knn_triage_threshold_all_points = 100*(1-triage_value)


    # apply knn to all points
    temp_points = pa.all_points[:,:2]
    #print ("temp points: ", temp_points.shape)
    idx_keep = knn_triage(knn_triage_threshold_all_points, temp_points)
    idx_keep = np.where(idx_keep==1)[0]
    all_points_knn = temp_points[idx_keep]

    # apply knn to lever/body movments
    triage_value = 0.1
    knn_triage_threshold_lever = 100*(1-triage_value)

    temp_points = pa.p_lever[0,:,:2]
    #print ("temp points: ", temp_points.shape)
    idx_keep = knn_triage(knn_triage_threshold_lever, temp_points)
    idx_keep = np.where(idx_keep==1)[0]
    p_lever_knn = temp_points[idx_keep]


    ################################################
    ###############################################
    ##############################################
    if plot_flag:
        if plot_3D:
            ax = fig.add_subplot(projection='3d')
        else:
            ax =plt.subplot(111)

    # clrs = ['red','pink','yellow']
    # cmap = matplotlib.cm.get_cmap('jet_r')

    # if plot_all:
    #     idx = np.arange(pa.all_points.shape[0])
    #     print (" all points: ", idx.shape)

    #print ("plever: ", p_lever_knn.shape)
    start = 0
    end = n_frames

   #for k in range(X_lever.shape[0]):
    if plot_flag:
        for k in range(start,end,1):
            if plot_3D:
                ax.scatter(p_lever_knn[k,:,0],
                           p_lever_knn[k,:,1],
                           p_lever_knn[k,:,2],
                        color=clr,
                        s=20,
                        edgecolor = 'black', alpha=.8)
            else:
                ax.scatter(p_lever_knn[:,0],
                           p_lever_knn[:,1],
                        color=clr,
                        s=20,
                        edgecolor = 'black', alpha=.8)

    #
    from scipy.spatial import ConvexHull

    # COMPUTE CONVEX HULL FOR ALL POINTS
    points = all_points_knn[:,:2]
    hull = ConvexHull(points)
    pa.points_simplex_all_points = points[hull.simplices]

    # COMPUTE CONVEX HULL FOR LEVER OR BODY PART
    points = p_lever_knn[:,:2]
    if points.shape[0]<3:
        pa.points_simplex = []
    else:
        hull = ConvexHull(points)
        pa.points_simplex = points[hull.simplices]

    ##############################
    if plot_flag:
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], c=clr)
        plt.plot(points[simplex[0], 0], points[simplex[-1], 1], c=clr)


        if plot_3D:
           if plot_all and pa.k==0:
                ax.scatter(all_points_knn [:,0],
                       all_points_knn [:,1],
                       all_points_knn [:,2],
                        c='black',
                        s=20,
                        edgecolor = 'black', alpha=.1)

           ax.scatter(p_lever_knn[:,0],
                       p_lever_knn[:,1],
                       p_lever_knn[:,2],
                        color=clr,
                        s=100,
                        edgecolor = 'black', alpha=.8)

        else:

           if plot_all and pa.k==0:
                ax.scatter(all_points_knn [:,0],
                       all_points_knn [:,1],
                        c='black',
                        s=100,
                        edgecolor = 'black', alpha=.1)

           ax.scatter(p_lever_knn[:,0],
                       p_lever_knn[:,1],
                        color=clr,
                        s=100,
                        edgecolor = 'black', alpha=.8)

    return pa

def plot_pca_scatter(pa,
                     n_frames = 30,
                     plot_all = True,
                     plot_3D=True):

    fig =plt.figure()
    if plot_3D:
        ax = fig.add_subplot(projection='3d')
    else:
        ax =plt.subplot(111)

    clrs = ['red','pink','yellow']
    cmap = matplotlib.cm.get_cmap('jet_r')

    idx = np.arange(pa.all_points.shape[0])
    print (idx.shape)

    print ("plever: ", pa.p_lever.shape)
    start = 0
    end = n_frames

    #
    #for k in range(X_lever.shape[0]):
    for k in range(start,end,1):
        if plot_3D:
            ax.scatter(pa.p_lever[k,:,0],
                       pa.p_lever[k,:,1],
                       pa.p_lever[k,:,2],
                    color=cmap(k/(end-start)),
                    s=20,
                    edgecolor = 'black', alpha=.8)
        else:
            ax.scatter(pa.p_lever[k,:,0],
                       pa.p_lever[k,:,1],
                    color=cmap(k/(end-start)),
                    s=20,
                    edgecolor = 'black', alpha=.8)

    if plot_3D:

        if plot_all:
            ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                   pa.all_points[idx,2],
                    c='black',
                    s=20,
                    edgecolor = 'black', alpha=.2)

        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                   pa.p_lever[0,:,2],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

    else:

        if plot_all:
            ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                    c='black',
                    s=100,
                    edgecolor = 'black', alpha=.8)

        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

    #
    # if False:
    #     plt.savefig('/home/cat/pca_all_plus_levers.png',dpi=300)
    #     plt.close()
    # else:
    #     plt.show()
    #                                    #

def plot_pca_scatter_lever_and_body_movements(pa, plot_3D=True):

    fig =plt.figure()
    if plot_3D:
        ax = fig.add_subplot(projection='3d')
    else:
        ax =plt.subplot(111)

    clrs = ['red','pink','yellow']
    cmap = matplotlib.cm.get_cmap('jet_r')

    idx = np.arange(pa.all_points.shape[0])

    print ("plever: ", pa.p_lever.shape)
    start = 0
    end = 30

    #
    # # plot lever dynamics in PCA/neural space
    # for k in range(start,end,1):
    #     if plot_3D:
    #         ax.scatter(pa.p_lever[k,:,0],
    #                    pa.p_lever[k,:,1],
    #                    pa.p_lever[k,:,2],
    #                    color=cmap(k/(end-start)),
    #                    s=20,
    #                    edgecolor = 'black', alpha=.8)
    #     else:
    #         ax.scatter(pa.p_lever[k,:,0],
    #                    pa.p_lever[k,:,1],
    #                 color=cmap(k/(end-start)),
    #                 s=20,
    #                 edgecolor = 'black', alpha=.8)

    # plot t=0 in neural space for lever and all_points
    if plot_3D:
        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                   pa.p_lever[0,:,2],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

        # plot all neural locationd
        ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                   pa.all_points[idx,2],
                    c='black',
                    s=20,
                    edgecolor = 'black', alpha=.2)
    else:
        ax.scatter(pa.p_lever[0,:,0],
                   pa.p_lever[0,:,1],
                    color='red',
                    s=100,
                    edgecolor = 'black', alpha=.8)

        ax.scatter(pa.all_points[idx,0],
                   pa.all_points[idx,1],
                    c='black',
                    s=100,
                    edgecolor = 'black', alpha=.8)

    if False:
        plt.savefig('/home/cat/pca_all_plus_levers.png',dpi=300)
        plt.close()
    else:
        plt.show()

def knn_triage(th, pca_wf):

    tree = cKDTree(pca_wf)
    dist, ind = tree.query(pca_wf, k=6)
    dist = np.sum(dist, 1)

    idx_keep1 = dist <= np.percentile(dist, th)
    return idx_keep1

def get_convex_hull(pa):


    # do very basic KNN triage
    n_dim = 3

    if True:

        from sklearn.neighbors import NearestNeighbors
        triage_value = 0.001
        knn_triage_threshold = 100*(1-triage_value)

        #if pca_wf.shape[0] > 1/triage_value:
        temp_points = pa.all_points[:,:n_dim]
        print ("temp points: ", temp_points.shape)
        idx_keep = knn_triage(knn_triage_threshold, temp_points)
        idx_keep = np.where(idx_keep==1)[0]

        print ("# points kept: ", idx_keep.shape,
               " of total: ", pa.all_points.shape[0])

        temp_points = temp_points[idx_keep]

    print ("computing convex hull # dim: ", n_dim, temp_points.shape)
    hull_all = ConvexHull(temp_points)

    #
    ratio_cumsum = []
    ratio_single = []
    ratio_random_single = []
    ratio_random_cumulative = []
    print ("p lever: ", pa.p_lever.shape)
    for k in trange(0, pa.p_lever.shape[0],1):

        # single frame convex hull
        points = pa.p_lever[k:k+1,:,:n_dim].squeeze()#.reshape(-1,3)
        try:
            hull11 = ConvexHull(points)
        except:
            continue
        ratio_single.append(hull11.volume/hull_all.volume)

        # cumulative convex hull
        points = pa.p_lever[:k+1,:,:n_dim].reshape(-1,n_dim)
        #if k<10:
        #    print ("cumulative: ", points.shape)
        hull1 = ConvexHull(points)
        ratio_cumsum.append(hull1.volume/hull_all.volume)

        #
        #idx = np.random.randint(0,temp_points.shape[0],p_lever.shape[1])
        id_ = np.random.randint(0,temp_points.shape[0]-500,1)
        idx = np.arange(id_,id_+pa.p_lever.shape[1])
        points_random = temp_points[idx,:n_dim]
        hull3 = ConvexHull(points_random)
        ratio_random_single.append(hull3.volume/hull_all.volume)

        #
        ratio_random_cumulative.append([])
        for q in range(10):
            if True:
                idx = np.random.randint(0,temp_points.shape[0],
                                        pa.p_lever.shape[1]*(k+1))
            else:
                ns = np.random.randint(0,temp_points.shape[0]-500,k+1)
                idx = []
                for s in range(ns.shape[0]):
                    idx.append(np.arange(ns[s],ns[s]+pa.p_lever.shape[1]))
                idx=np.hstack(idx)
            points_random = temp_points[idx]
            #if q==0 and k<10:
            #    print (points_random.shape)
            hull4 = ConvexHull(points_random)
            ratio_random_cumulative[k].append(hull4.volume/hull_all.volume)

    pa.ratio_single = np.array(ratio_single)[::-1]
    pa.ratio_cumsum = np.array(ratio_cumsum)[::-1]
    pa.ratio_random_single = np.array(ratio_random_single)[::-1]
    pa.ratio_random_cumulative = np.array(ratio_random_cumulative)[::-1]

    print (pa.ratio_random_cumulative.shape)
    print (pa.ratio_random_single.shape)

    fname_out = os.path.join(pa.root_dir,
                             pa.animal_id,
                             'tif_files',
                             pa.session,
                             pa.session+"_convex_hull.npz")

    np.savez(fname_out,
             ratio_single = ratio_single,
             ratio_cumsum = ratio_cumsum,
             ratio_random_single = ratio_random_single,
             ratio_random_cumulative = ratio_random_cumulative
             )

    return pa


def plot_convex_hull(pa):

    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############
    plt.scatter(t, pa.ratio_cumsum,
               s=100,edgecolor='black',
               c='red', label = 'Cumulative ConvexHull',
               alpha=.8)

    # vs random cumulative
    mean = np.mean(pa.ratio_random_cumulative,1)
    std = np.std(pa.ratio_random_cumulative,1)

    plt.fill_between(t, mean+std, mean-std,
                     color='red',
                     alpha=.2)

    plt.plot(t,mean,'--',
             linewidth=4, label = 'Cumulative ConvexHull - Random',
            c='red')

    ############# SINGLE############
    plt.scatter(t, pa.ratio_single,
               s=100,edgecolor='black', label= "ConvexHull",
               c='blue',
               alpha=.8)

    # vs random single
    mean = np.mean(pa.ratio_random_single)
    std = np.std(pa.ratio_random_single)
    plt.fill_between(t, mean+std, mean-std,
                     color='blue',
                    alpha=.2)

    plt.plot([t[0],t[-1]],[mean,mean],'--',
             label= "ConvexHull - Random",
              linewidth=4,

            c='blue')

    plt.legend(*(
        [ x[i] for i in [3,1,2,0] ]
        for x in plt.gca().get_legend_handles_labels()
    ), handletextpad=0.75, loc='best',
              fontsize=20)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()



def plot_convex_hull_cumulative_only(pa, cmap, clr_ctr,
                                     alpha):


    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############

    #plt.scatter(t, pa.ratio_cumsum,
    ax1=plt.subplot(121)
    plt.scatter(t, pa.ratio_cumsum,
               s=100,edgecolor='black',
               color=cmap(clr_ctr),
               label = 'Cumulative ConvexHull',
               alpha=alpha)
    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.xticks([])
    plt.yticks([])

    ax2=plt.subplot(122)
    plt.scatter(t, pa.ratio_cumsum/np.max(pa.ratio_cumsum),
               s=100,edgecolor='black',
               color=cmap(clr_ctr),
               label = 'Cumulative ConvexHull',
               alpha=alpha)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.xticks([])
    plt.yticks([])

#
def plot_convex_hull_single_only(pa, cmap, clr_ctr, alpha):


    t = np.arange(pa.ratio_single.shape[0])/30.-10

    ########### CUMSUM ############
    mean = np.mean(pa.ratio_random_single)

    #plt.scatter(t, pa.ratio_cumsum,
    ax1=plt.subplot(121)
    plt.scatter(t, pa.ratio_single,
               s=100, edgecolor='black', label= "ConvexHull",
               color=cmap(clr_ctr),
               alpha=alpha)
    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()

    ax2=plt.subplot(122)
    plt.scatter(t, pa.ratio_single/np.max(pa.ratio_single),
               s=100,edgecolor='black', label= "ConvexHull",
               color=cmap(clr_ctr),
               alpha=alpha)

    plt.ylim(0,1)
    plt.xlim(t[0],t[-1])
    plt.show()

# load triggers for each label
def load_starts_body_movements(pa
                              ):

    fname = os.path.join(pa.root_dir,
                        pa.animal_id,
                         'tif_files',
                         pa.session_corrected,
                         pa.session_corrected+"_3secNoMove_movements.npz"
                        )

    try:
        data = np.load(fname, allow_pickle=True)
    except:
        print (" No video/3sec movement files...")
        return []

    features = data['feature_quiescent']
    #print (features.shape)
    #labels = data['labels']
    #print (labels)
    #temp = np.vstack(features[0])

    starts = []
    for k in range(len(features)):
        temp = features[k]
        if len(temp)>0:
            #print (temp)
            starts.append(np.vstack(temp)[:,1])
        else:
            starts.append([])

    return starts


# Load the manual aligned shifts for each animal

def load_shift_ids(pa):

    fnames_good = '/media/cat/4TBSSD/yuki/'+pa.animal_id + '/tif_files/sessions_DLC_alignment_good.txt'

    import csv
    sessions = []
    shift_ids = []
    with open(fnames_good, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            sessions.append(str(row[0]).replace(',',''))
            shift_ids.append(row[1])

    #print ("SESSIONS: ", sessions)
    #print ("SHIFT IDS: ", shift_ids)
    #print ("pa.session_id: ", pa.session_id)

    # grab the session id from the ordered data to figure out correct ID to shift
    found = False
    for k in range(len(sessions)):
        # print ("sessions[k]", sessions[k])
        # print ("pa.sessoin_idd: ", pa.session_id)
        if sessions[k] in pa.session_id:
            ctr_plt = k
            found = True
            break
        # if pa.session_id in sessions[k]:
        #     ctr_plt = k
        #     found = True
        #     break

    if found:
        shift_id_str = shift_ids[ctr_plt]
    else:
        shift_id_str = None

    return shift_id_str


def load_shifts(pa):
    # load the DLC correlation shift
    fname_correlate = os.path.join(pa.root_dir, pa.animal_id,
                         'tif_files', pa.session,
                         'correlate.npz')

    try:
        data = np.load(fname_correlate,allow_pickle=True)
    except:
        print( " ... data missing", fname_correlate)
        asdf

    cors = data['cors'].squeeze().T

    #
    #print ("SELF SHIFT ID: ", pa.shift_id_str)
    if len(pa.shift_id_str)>1:
        pa.shift_id = int(pa.shift_id_str[0])
        pa.shift_additional = float(pa.shift_id_str[1:])
    else:
        pa.shift_id = int(pa.shift_id_str)
        pa.shift_additional = 0

    #print ( " using shift: ", pa.shift_id+pa.shift_additional)

    corr_featur_id = pa.shift_id

    temp_trace = cors[:,corr_featur_id]
    temp_trace[:2000] = 0
    temp_trace[-2000:] = 0
    DLC_shift = round(np.argmax(temp_trace)/1000. - 15.,2)+pa.shift_additional
    #print ("DLC SHIFT Loaded: ", DLC_shift)

    return DLC_shift


#
def pca_scatter_body_movements_fig3(pa, sessions):


    res_simplex = []
    for session in sessions:

        pa.session_id = session
        pa.session_corrected = session
        pa.session = session
        #print ("pa.session_id 1: ", pa.session_id )
        #print("session corrected: ", pa.session_corrected)

        #
        fname_simplex = os.path.join(pa.root_dir,pa.animal_id,'tif_files',pa.session,
                                 pa.session+ "_simplex.npy")

        if os.path.exists(fname_simplex)==False:
            ################################
            # load body movement starts
            starts = load_starts_body_movements(pa)
            if len(starts)==0:
                res_simplex.append([])
                continue

            #print ("pa.session_id 2: ", pa.session_id )
            #print ("starts: ", starts[0][:10])

            ################################
            # get the lever offset
            lever_offset = get_lever_offset(pa.root_dir,
                                            pa.animal_id,
                                            pa.session)
            #print ("lever offset: ", lever_offset)

            ###################################
            # load shift ids
            #pa.session_id = pa.session
            pa.shift_id_str = load_shift_ids(pa)
            if pa.shift_id_str is None:
                res_simplex.append([])
                continue

            ##################################
            # load shift
            DLC_shift = load_shifts(pa)
            shift_frames = int(DLC_shift*pa.frame_rate)
            shift_relative = -(shift_frames-lever_offset)
            #print ("DLC shift : ", DLC_shift,
            #       "  in frames: ", shift_frames,
            #       " after subtracting lever shift: ", shift_relative)

            ####################################
            # visualize
            clrs=['black','magenta','blue','green','red','yellow']

            # first load the raw data and lever triggers
            pa.X, pa.triggers = get_data_and_triggers(pa)
            if pa.X is None:
                res_simplex.append([])
                continue

            # get PCA or UMAP object and all points;
            pa.pca, pa.all_points = get_pca_object_and_all_points(pa)
            #print (" pca allpoints: ", pa.all_points.shape)

            ## use dim reduced data as input to UMAP
            if pa.umap_flag:
                pa.umap, pa.all_points = get_umap_object_and_all_points(pa)
                #print (" umap allpoints: ", pa.all_points.shape)

            #
            # loop over all features
            pa_array = []
            pa_simplex = []

            if pa.plot_flag:
                fig = plt.figure(figsize=(5,5))

            for k in range(5):
                pa.k=k

                #############################
                ######### PLOT LIMBS ########
                #############################
                pa.triggers = np.int32(np.array(starts[k])*pa.frame_rate) + shift_relative
                #print ("pa triggers: ", pa.triggers)

                if pa.triggers.shape[0]==0:
                    pa_array.append([])
                    pa_simplex.append([])
                    continue

                # FIG 2 E top
                # get pca projection first
                pa.p_lever =  project_data_pca(pa)
                #print ("  body movement pa.p_lever projection: ",
                #       pa.p_lever.shape)

                # get UMAP projection second
                if pa.umap_flag:
                    pa.p_lever = project_data_umap(pa)

                pa.plot_all = True

                #
                pa = plot_pca_scatter_multi(pa,
                                 n_frames=pa.n_frames,
                                 clr = clrs[k],
                                 plot_all=pa.plot_all,
                                 plot_3D = pa.plot_3D,
                                 plot_flag = pa.plot_flag)

                pa_array.append(pa.p_lever)
                pa_simplex.append(pa.points_simplex)

            #############################
            ######### PLOT LEVER ########
            #############################
            fname_lever = os.path.join(pa.root_dir, pa.animal_id,'tif_files',pa.session, pa.session+"_all_locs_selected.txt")
            #print ("FNAME LEVER: ", fname_lever)
            lever_times =np.loadtxt(fname_lever)
            pa.triggers = np.int32(np.array(lever_times)*pa.frame_rate)
            # print ("pa triggers: ", pa.triggers)

            #
            pa.p_lever =  project_data_pca(pa)

            # get UMAP projection second
            if pa.umap_flag:
                pa.p_lever =  project_data_umap(pa)

            pa.plot_all = True
            pa = plot_pca_scatter_multi(pa,
                             n_frames=pa.n_frames,
                             clr = 'brown',
                             plot_all=pa.plot_all,
                             plot_3D = pa.plot_3D,
                             plot_flag = pa.plot_flag)

            pa_simplex.append(pa.points_simplex)

            ########################################
            ######### ADD ALL POINTS TO END ########
            ########################################

            pa_simplex.append(pa.points_simplex_all_points)

            #
            if pa.plot_flag:
                if pa.umap_flag==False:
                    plt.ylim(-40000,60000)
                    plt.xlim(-60000,60000)
                plt.xticks([])
                plt.yticks([])

                plt.title(pa.session_corrected)

                if True:
                    plt.savefig('/home/cat/umap.png',dpi=300)
                    plt.close()
                else:
                    plt.show()

            res_simplex.append(pa_simplex)

            np.save(fname_simplex, pa_simplex)

        else:
            pa_simplex = np.load(fname_simplex, allow_pickle=True)
            res_simplex.append(pa_simplex)

    return res_simplex



def convexhull(p):
    p = np.array(p)
    hull = ConvexHull(p)
    return p[hull.vertices,:]



def plot_intersection_convex_hulls_lever_vs_bodyparts(res_simplex,
                                                      sessions,
                                                      animal_id,
                                                      root_dir):
    from shapely import geometry

    names = ['leftpaw','rightpaw','nose','jaw','ear','lever']
    clrs=['black','magenta','blue','green','red','brown']


    ctr = 0
    lever_vs_left_paw = []
    lever_vs_right_paw = []
    lever_vs_all = []
    n_trials = []
    for k in range(len(res_simplex)):

        # find # of lever pulls in sessions
        fname_session = os.path.join(root_dir, animal_id,'tif_files',sessions[k],
                                     sessions[k]+"_all_locs_selected.txt")


        #
        pa_simplex = res_simplex[k]
        if len(pa_simplex)==0:
            continue

        #
        if os.path.exists(fname_session):
            lever_pull_times = np.loadtxt(fname_session)
            n_pulls = lever_pull_times.shape[0]
        else:
            n_pulls = np.nan


        n_trials.append(n_pulls)

        if ctr<=45:
            ax=plt.subplot(5,10,ctr+1)

        ############################################
        ### PLOT POLYGONS FOR EACH BODY MOVEMENT ###
        ############################################
        polygons = []
        for p in range(len(pa_simplex)-1):
            pol_temp = []
            if len(pa_simplex[p])>0:
                for aa in range(pa_simplex[p].shape[0]):
                    temp = pa_simplex[p][aa].T

                    if ctr<=45:
                        # only use label once
                        if aa==0:
                            plt.plot(temp[0],
                                 temp[1],c=clrs[p],
                                 label=names[p],
                                 linewidth=5)
                        else:
                            plt.plot(temp[0],
                                 temp[1],
                                 c=clrs[p],
                                 linewidth=5)

                    pol_temp.append(temp.T[0])
                    pol_temp.append(temp.T[1])
                pol_temp = np.vstack(pol_temp)
                pol_temp = np.unique(pol_temp,axis=0)
                pol_temp = convexhull(pol_temp)
                polygons.append(pol_temp)
            else:
                polygons.append([])

        ########################################
        ### Compute all_points outer surface ###
        ########################################
        pts_simplex_all_points = pa_simplex[-1]
        pol_temp =[]
        for a in range(pts_simplex_all_points.shape[0]):
            temp = pts_simplex_all_points[a].T
            if a==0:
                plt.plot(temp[0],
                     temp[1],
                     c='grey',
                     label = 'all',
                    linewidth=5)
            else:
                plt.plot(temp[0],
                     temp[1],
                     c='grey',
                    linewidth=5)

            pol_temp.append(temp.T[0])
            pol_temp.append(temp.T[1])

        pol_temp = np.vstack(pol_temp)
        pol_temp = np.unique(pol_temp,axis=0)
        pol_temp = convexhull(pol_temp)
        polygons_all = pol_temp
        #

        ########################################
        # Compute areas of all neural spaces ###
        ########################################
        areas = []
        for p in range(len(polygons)):
            if len(polygons[p])>0:
                polygon1 = geometry.Polygon(polygons[p])
                polygon2 = geometry.Polygon(polygons[p])
                areas.append(polygon1.intersection(polygon2).area)
            else:
                areas.append(np.nan)

        print ("all areas: ", areas)
        polygon1 = geometry.Polygon(polygons_all)
        polygon2 = geometry.Polygon(polygons_all)
        area_all = polygon1.intersection(polygon2).area

        ############################################
        ##### COMPUTE Intersection lever and all ###
        ############################################
        polygon1 = geometry.Polygon(polygons[5])
        polygon2 = geometry.Polygon(polygons_all)

        intersection = polygon1.intersection(polygon2).area
        lever_vs_all.append(intersection/area_all)

        ####################################
        ###### compute intersections #######
        ####################################
        from shapely import geometry, ops
        for q in range(len(polygons)):
            for p in range(len(polygons)):
                if q==p:
                    continue

                polygon1 = geometry.Polygon(polygons[q])
                polygon2 = geometry.Polygon(polygons[p])

                intersection = polygon1.intersection(polygon2).area

                #
                if q == 0 and p == 5:
                    if np.isnan(areas[p])==False:
                        ratio = intersection/areas[q]
                        #ratio = ratio/n_pulls
                        lever_vs_left_paw.append(ratio)

                        print (names[q],names[p],
                           "% of region: ", round(ratio*100,2), "%",
                           "% of all space: ", round(intersection/area_all*100,2), "%")
                    else:
                        lever_vs_left_paw.append(np.nan)

                elif q == 1 and p == 5:
                    if np.isnan(areas[p])==False:
                        ratio = intersection/areas[q]
                        #ratio = ratio/n_pulls

                        lever_vs_right_paw.append(ratio)
                        print (names[q],names[p],
                           "% of region: ", round(ratio*100,2), "%",
                           "% of all space: ", round(intersection/area_all*100,2), "%")
                    else:
                        lever_vs_right_paw.append(np.nan)

            print ('')
        plt.xticks([])
        plt.yticks([])

        plt.title(sessions[k])

        #
        ctr+=1

    return lever_vs_left_paw, lever_vs_right_paw, lever_vs_all, n_trials
