import matplotlib
import matplotlib.cm as cm
from matplotlib import gridspec
import parmap
import numpy as np
import os
import matplotlib.pyplot as plt

import glob2
import scipy

from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Visualize():

    def __init__(self):

        self.clr_ctr = 0

        #
        self.animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        #
        self.colors = ['black','blue','red','green','magenta','pink','cyan']

        #
        self.n_colors = [10,19,23,17,20,48]

        #
        self.filter=False

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
        #print (self.sessions)

        data = []
        for k in range(len(self.sessions)):
            data.append(os.path.split(self.sessions[k])[1][:-4])
        self.sessions = data

        #
        final_session = []
        for k in range(len(self.sessions)):
            if self.session_id in self.sessions[k]:
                final_session = self.sessions[k]
                break

        self.session = final_session

        print ("Self session:", self.session)


        prefix1 = ''
        if self.lockout:
            prefix1 = '_lockout_'+str(self.lockout_window)+"sec"
        #
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

    def plot_decision_choice(self, clr, label, title=None, ax=None):

        #
        if ax is None:
            ax=plt.subplot(111)

        self.get_fname()


        self.data = np.load(self.fname)
            # vis.load_data(fname)


        #
        mean = self.data.mean(1)
        t = np.linspace(-9.5, 9.5, mean.shape[0])
        std = np.std(self.data,axis=1)
        ax.plot(t, mean,
                 c=clr,
                 label = label,
                 linewidth=4)
        ax.fill_between(t, mean-std, mean+std, alpha = 0.2)

        self.format_plot(ax)

        plt.legend(fontsize=20)
        if title is not None:
            plt.suptitle(title)



    def plot_significant(self, clr, label, animal_id, session):


        #
        fig=plt.figure()
        ax=plt.subplot(111)


        #
        mean = self.data.mean(1)
        t = np.linspace(-9.5, 9.5, mean.shape[0])
        std = np.std(self.data,axis=1)
        plt.plot(t, mean,
                 c=clr,
                 label = label,
                 linewidth=4)
        plt.fill_between(t, mean-std, mean+std, alpha = 0.2)


        # grab end points
        control = np.hstack((self.data[-2:].flatten(),
                             self.data[:2].flatten()))

        #control = self.data[:3].flatten()
        print ("controL ", control.shape)

        sig = []
        for k in range(self.data.shape[0]):
            res = stats.ks_2samp(self.data[k],
                                 control)

            #res = stats.ttest_ind(first, second, axis=0, equal_var=True)
            res = scipy.stats.ttest_1samp(self.data[k], 0.5)

            sig.append(res[1])

        sig=np.array(sig)[None]
        print (sig)
        thresh = 0.05
        idx = np.where(sig>thresh)
        sig[idx]=np.nan
        vmin = np.nanmin(sig)
        vmax = np.nanmax(sig)
        vmin=0.0
        vmax=0.05

        axins = ax.inset_axes((0,.95,1,.05))
        axins.set_xticks([])
        axins.set_yticks([])
        im = axins.imshow(sig,vmin=vmin,vmax=vmax,
                          aspect='auto',
                         cmap='viridis_r')
        #
        ticks = np.round(np.linspace(vmin, vmax, 4),3)
        print ("vmin, vmax; ", vmin, vmax, "ticks: ", ticks)
        fmt = '%1.3f'
        #
        cbar = fig.colorbar(im, ax=ax, shrink=0.4,
                            ticks=ticks, format = fmt)

        cbar.ax.tick_params(labelsize=16)
        cbar.ax.set_title('Pval', rotation=0,
                         fontsize=16)

        self.format_plot(ax)
        plt.title(animal_id + "  session: "+str(session))
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
        animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=animal_names.index(animal_name)

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


    def plot_animal_time_longitudinal(self, animal_name):
        animal_names = ['IA1','IA2','IA3','IJ1','IJ2','AQ2']

        idx=animal_names.index(animal_name)

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
        for animal_name in self.animal_names:
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
                    label=self.animal_names[self.clr_ctr])

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
        idx=self.animal_names.index(animal_name)

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
            root_dir = self.main_dir+self.animal_names[a]+'/'
            fnames = np.sort(glob2.glob(root_dir+'conf_10_'+self.animal_names[a]+"*"))

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
            axes[a].set_title(self.animal_names[a])
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


    def filter_trace(self,trace,width=1):

        box = np.ones(width)/width
        trace_smooth = np.convolve(trace, box, mode='same')

        return trace_smooth
