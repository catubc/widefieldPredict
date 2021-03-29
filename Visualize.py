import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import os
import matplotlib.pyplot as plt

import glob2
import scipy
from tqdm import trange

from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Visualize():

    def __init__(self):

        self.clr_ctr = 0

        #
        self.animal_ids = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        #
        self.colors = ['black','blue','red','green','magenta','pink','cyan']

        #
        self.n_colors = [10,19,23,17,20,48]

        #
        self.linewidth = 4

        #
        self.filter=False

        #
        self.cbar_offset = 0


    def load_data(self, fname):

        self.data = np.load(fname)



    def format_plot(self, ax):
        ''' Formats plots for decision choice with 50% and 0 lines
        '''
       # meta data
        xlims = [-10,10]
        ylims = [0.4,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 'r--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.5,0.5],
                 'r--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(-10,10)
        ax.set_ylim(0.4,1.0)



    def format_plot2(self, ax):
        ''' Format plots for time prediction
        '''
       # meta data
        xlims = [-10,0]
        ylims = [0.0,1.0]
        ax.plot([0,0],
                 [ylims[0],ylims[1]],
                 'r--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.plot([xlims[0],xlims[1]],
                 [0.1,0.1],
                 'r--',
                linewidth=3,
                 color='black',
                alpha=.5)

        ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
       # plt.legend(fontsize=20)


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

        # select data with or without lockout
        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"

        # select data with pca compression
        prefix2 = ''
        if self.pca_flag:
            prefix2 = '_pca_'+str(self.pca_var)

        # make fname out for animal + session
        fname = os.path.join(self.main_dir, self.animal_id,
                             'SVM_Scores',
                             'SVM_Scores_'+
                             self.session+"_"+
                             self.code+
                             prefix1+
                             '_trial_ROItimeCourses_'+
                             str(self.window)+'sec'+
                             prefix2+
                             '.npy'
                             )
        self.fname = fname

    def get_number_of_trials(self):

        fname_txt = os.path.join(self.main_dir,
                                 self.animal_id,
                                 'tif_files',
                                 self.session_id,
                                 self.session_id+"_all_locs_selected.txt")

        if os.path.exists(fname_txt)==False:
            self.n_trials = 0
            self.n_trials_lockout = 0
            return

        self.n_trials = np.loadtxt(fname_txt).shape[0]

        fname_txt = os.path.join(self.main_dir,
                                 self.animal_id,
                                 'tif_files',
                                 self.session_id,
                                 self.session_id+"_lockout_10sec_locs_selected.txt")
        self.n_trials_lockout = np.loadtxt(fname_txt).shape[0]

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



    def process_session(self):

        # get n trials for both lockout and all trials data
        self.get_number_of_trials()

        #
        if self.n_trials<self.min_trials:
            self.data = np.zeros((0))
            return

        # gets the corect filename to be loaded below
        self.get_fname()

        #
        if os.path.exists(self.fname)==False:
            self.data = np.zeros((0))
            return
        self.data = np.load(self.fname)

        #
        mean = self.data.mean(1)
        #
        if self.smooth_window is not None:
            mean = self.filter_trace(mean)
            data = []
            for k in range(self.data.shape[1]):
                data.append(self.filter_trace(self.data[:,k]))
            self.data = np.array(data).copy().T

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

        sig = []
        for k in range(self.data.shape[0]):
            #res = stats.ks_2samp(self.data[k],
            #                     control)
            #res = stats.ttest_ind(first, second, axis=0, equal_var=True)

            #
            res = scipy.stats.ttest_1samp(self.data[k], 0.5)

            sig.append(res[1])

        sig=np.array(sig)[None]
        thresh = self.significance
        idx = np.where(sig>thresh)
        sig[idx] = np.nan

        #
        idx = np.where(self.mean<0.5)
        sig[:,idx] = np.nan

        #
        self.sig =sig[:,:sig.shape[1]//2]

    def compute_first_decoding_time(self):

        #
        lockouts = [False, True]
        for lockout in lockouts:
            self.lockout=lockout

            all_res_continuous = []
            all_res_earliest = []
            times = []

            #
            for a in trange(len(self.animal_ids)):
                res_continuous = []
                res_earliest = []
                t = []

                #
                self.animal_id = self.animal_ids[a]
                self.get_sessions()

                #
                for p in range(len(self.session_ids)):
                    self.session_id = self.session_ids[p]
                    self.process_session()
                    #
                    if self.data.shape[0] == 0:
                        continue

                    # compute significance
                    self.compute_significance()
                    self.sig = self.sig.squeeze()

                    # find continous period earliest
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            break

                    temp = -10+k/30.

                    # Exclude one of the weird datapoints
                    if temp>0:
                        #print ("n trials: ", self.n_trials, a,
                        #       p, temp, self.session_id, self.sig.shape)
                        continue
                    res_continuous.append(temp)

                    # find aboslute earliest
                    k_earliest = self.sig.shape[0]
                    for k in range(self.sig.shape[0]-1,0,-1):
                        if np.isnan(self.sig[k])==True:
                            k_earliest = k
                    res_earliest.append(-10+k_earliest/30.)

                    #
                    t.append(p)

                # save data
                all_res_continuous.append(res_continuous)
                all_res_earliest.append(res_earliest)
                times.append(t)

            if lockout==False:
                np.save('/home/cat/all_cont.npy', all_res_continuous)
                np.save('/home/cat/all_earliest.npy', all_res_earliest)
                np.save('/home/cat/times.npy', times)

            else:
                np.save('/home/cat/all_cont_lockout.npy', all_res_continuous)
                np.save('/home/cat/all_earliest_lockout.npy', all_res_continuous)


    def plot_first_decoding_time(self):
        labels = ["M1", "M2", "M3", "M4","M5",'M6']

        # flag to search for any signfiicant decoding time, not just continous ones
        earliest = False

        if earliest==False:
            all_res_continuous_all = np.load('/home/cat/all_cont.npy', allow_pickle=True)
            all_res_continuous_lockout = np.load('/home/cat/all_cont_lockout.npy', allow_pickle=True)
        else:
            all_res_continuous_all = np.load('/home/cat/all_earliest.npy', allow_pickle=True)
            all_res_continuous_lockout = np.load('/home/cat/all_earliest_lockout.npy', allow_pickle=True)

        #
        data_sets_all = []
        for k in range(len(all_res_continuous_all)):
            data_sets_all.append(all_res_continuous_all[k])
        #
        data_sets_lockout = []
        for k in range(len(all_res_continuous_lockout)):
            data_sets_lockout.append(all_res_continuous_lockout[k])

        # Computed quantities to aid plotting

        hist_range = (-10,0)
        bins = np.arange(-10,0,1)

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
            ax.barh(centers, -binned_data, height=heights, left=lefts, color='black')

            lefts = x_loc #- 0.5 * binned_data_lockout
            ax.barh(centers, binned_data_lockout, height=heights, left=lefts, color='blue')

        ax.set_xticks(x_locations)
        ax.set_xticklabels(labels)
        ax.set_xlim(-20,spacing*6)
        ax.set_ylim(-10.5,0)


        ax.set_ylabel("Data values")
        ax.set_xlabel("Data sets")


    def plot_significant(self, fig, clr, label):

        #
        ax=plt.subplot(111)

        # load data
        self.get_fname()
        self.data = np.load(self.fname)

        #
        self.process_session()

        #
        t = np.linspace(-9.5, 9.5, self.mean.shape[0])

        # plotting
        plt.plot(t,
                 self.mean,
                 c=clr,
                 label = label,
                 linewidth=self.linewidth)
        plt.fill_between(t, self.mean-self.std, self.mean+self.std, color=clr, alpha = 0.2)

        # compute significance
        self.compute_significance()

        # set
        vmin=0.0
        vmax=self.significance

        # plot significance
        axins = ax.inset_axes((0,.95-self.cbar_offset,1,.05))
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
        cbar = fig.colorbar(im,
                            ax=ax,
                            shrink=0.2,
                            ticks=ticks,
                            format = fmt)

        cbar.ax.tick_params(labelsize=25)
        #cbar.ax.set_title('Pval',
                          #rotation=0,
                          #pad=1.8,
                          #fontsize=25)

        self.format_plot(ax)
        plt.title(self.animal_id + "  session: "+str(self.session))

        #
        self.cbar_offset+=0.05

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

        self.fnames_svm = []
        self.session_ids = []
        for k in range(len(self.sessions)):
            self.session = os.path.split(self.sessions[k])[1][:-4]
            self.session_ids.append(self.session)

            # save full path of svm data
            fname = os.path.join(self.main_dir, self.animal_id,
                     'SVM_Scores',
                     'SVM_Scores_'+
                     self.session+"_"+
                     self.code+
                     prefix1+
                     '_trial_ROItimeCourses_'+
                     str(self.window)+'sec'+
                     prefix2+
                     '.npy'
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
                     'r--', linewidth=4,
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


