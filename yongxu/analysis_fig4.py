import matplotlib

#
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
import torch
import time
import warnings
warnings.filterwarnings("ignore")
from locanmf import LocaNMF
import postprocess

import os

device='cpu'


class LocaNMFClass():

    def __init__(self, root_dir, animal_id, session):

        #
        self.min_trials = 10

        #
        self.root_dir = root_dir

        #
        self.animal_id = animal_id   # 'IJ1'

        #
        self.sessions = self.get_sessions(session)     # 'Mar3'

        #
        fname_atlas = os.path.join(self.root_dir, 'yongxu/atlas_split.npy')
        self.atlas = np.load(fname_atlas)



    def get_sessions(self,session_id):
         # load ordered sessions from file
        sessions = np.load(os.path.join(self.root_dir,
                                        self.animal_id,
                                        'tif_files.npy'))
        # grab session names from saved .npy files
        data = []
        for k in range(len(sessions)):
            data.append(os.path.split(sessions[k])[1].replace('.tif',''))
        sessions = data

        #
        if session_id != 'all':
            final_session = []
            session_number = None
            for k in range(len(sessions)):
                if session_id in sessions[k]:
                    final_session = [sessions[k]]
                    session_number = k
                    break
            sessions = final_session

        # fix binary string files issues; remove 'b and ' from file names
        for k in range(len(sessions)):
            sessions[k] = str(sessions[k]).replace("'b",'').replace("'","")
            if sessions[k][0]=='b':
                sessions[k] = sessions[k][1:]

        sessions = np.array(sessions)

        return sessions

    def run_loca(self):

        #################################################
        #################################################
        #################################################
        # maxrank = how many max components per brain region. Set maxrank to around 4 for regular dataset.
        maxrank = 1

        # min_pixels = minimum number of pixels in Allen map for it to be considered a brain region
        # default min_pixels = 100
        min_pixels = 200

        # loc_thresh = Localization threshold, i.e. percentage of area restricted to be inside the 'Allen boundary'
        # default loc_thresh = 80
        loc_thresh = 75

        # r2_thresh = Fraction of variance in the data to capture with LocaNMF
        # default r2_thresh = 0.99
        r2_thresh = 0.96

        # Do you want nonnegative temporal components? The data itself should also be nonnegative in this case.
        # default nonnegative_temporal = False
        nonnegative_temporal = False


        # maxiter_hals = Number of iterations in innermost loop (HALS). Keeping this low provides a sort of regularization.
        # default maxiter_hals = 20
        maxiter_hals = 20

        # maxiter_lambda = Number of iterations for the lambda loop. Keep this high for finding a good solution.
        # default maxiter_lambda = 100
        maxiter_lambda = 150

        # lambda_step = Amount to multiply lambda after every lambda iteration.
        # lambda_init = initial value of lambda. Keep this low. default lambda_init = 0.000001
        # lambda_{i+1}=lambda_i*lambda_step. lambda_0=lambda_init. default lambda_step = 1.35
        lambda_step = 1.25
        lambda_init = 1e-4

        ######################################################
        ######################################################
        ######################################################
        for session in self.sessions:

            fname_out = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                  session,session+'_locanmf.npz')
            if os.path.exists(fname_out)==False:

                fname_locs = os.path.join(self.root_dir, self.animal_id, 'tif_files',
                                          session, session + '_all_locs_selected.txt')
                if os.path.exists(fname_locs)==False:
                    print ("  no lever pulls, skipping ")
                    continue

                n_locs = np.loadtxt(fname_locs)
                print ("")
                print ("")
                print (session, " has n trials: ", n_locs.shape)
                if n_locs.shape[0]<self.min_trials:
                    print ("  too few trials, skipping ", n_locs.shape[0])
                    continue

                fname_spatial = os.path.join(self.root_dir,self.animal_id, 'tif_files',
                                             session,
                                             #session+'_code_04_trial_ROItimeCourses_15sec_pca_0.95_spatial.npy')
                                             session+'_code_04_trial_ROItimeCourses_30sec_pca_0.95_spatial.npy')

                spatial = np.load(fname_spatial)
                spatial = np.transpose(spatial,[1,0])
                denoised_spatial_name = np.reshape(spatial,[128,128,-1])
                # print ("denoised_spatial_name: ", denoised_spatial_name.shape)
                #
                temporal_trial = np.load(fname_spatial.replace('_spatial',''))

                #
                temporal_random = np.load(fname_spatial.replace('trial','random').replace('_spatial',''))

                # make sure there are same # of trials in random and trial dataset
                min_trials = min(temporal_trial.shape[0], temporal_random.shape[0])
                temporal_trial = temporal_trial[:min_trials]
                temporal_random = temporal_random[:min_trials]

                #
                temporal=np.concatenate((temporal_trial,temporal_random),axis=0)
                temporal = np.transpose(temporal,[1,0,2])

                denoised_temporal_name = np.reshape(temporal,[-1,temporal.shape[1]*temporal.shape[2]])
                #print('loaded data',flush=True)

                #######################################
                # Get data in the correct format
                V=denoised_temporal_name
                U=denoised_spatial_name

                #
                brainmask = np.ones(U.shape[:2],dtype=bool)

                # Load true areas if simulated data
                simulation=0

                # Include nan values of U in brainmask, and put those values to 0 in U
                brainmask[np.isnan(np.sum(U,axis=2))]=False
                U[np.isnan(U)]=0

                # Preprocess V: flatten and remove nans
                dimsV=V.shape
                keepinds=np.nonzero(np.sum(np.isfinite(V),axis=0))[0]
                V=V[:,keepinds]

                #
                if V.shape[0]!=U.shape[-1]:
                    print('Wrong dimensions of U and V!')

                print("Rank of video : %d" % V.shape[0])
                print("Number of timepoints : %d" % V.shape[1]);


                ##################################################
                ##################################################
                ##################################################
                # Perform the LQ decomposition. Time everything.
                t0_global = time.time()
                t0 = time.time()
                if nonnegative_temporal:
                    r = V.T
                else:
                    q, r = np.linalg.qr(V.T)
                # time_ests={'qr_decomp':time.time() - t0}

                # Put in data structure for LocaNMF
                video_mats = (np.copy(U[brainmask]), r.T)
                rank_range = (1, maxrank, 1)
                del U


                ##################################################
                ##################################################
                ##################################################

                #
                region_mats = LocaNMF.extract_region_metadata(brainmask,
                                                              self.atlas,
                                                              min_size=min_pixels)

                #
                region_metadata = LocaNMF.RegionMetadata(region_mats[0].shape[0],
                                                         region_mats[0].shape[1:],
                                                         device=device)

                #
                region_metadata.set(torch.from_numpy(region_mats[0].astype(np.uint8)),
                                    torch.from_numpy(region_mats[1]),
                                    torch.from_numpy(region_mats[2].astype(np.int64)))


                ##################################################
                ##################################################
                ##################################################

                # grab region names
                rois=np.load('./rois_50.npz')
                rois_name=rois['names']

                rois_ids=rois['ids']

                ##################################################
                ##################################################
                ##################################################

                # Do SVD as initialization
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v SVD Initialization')
                t0 = time.time()
                region_videos = LocaNMF.factor_region_videos(video_mats,
                                                             region_mats[0],
                                                             rank_range[1],
                                                             device=device)
                #
                if device=='cuda':
                    torch.cuda.synchronize()
                print("\'-total : %f" % (time.time() - t0))
                #time_ests['svd_init'] = time.time() - t0


                #
                low_rank_video = LocaNMF.LowRankVideo(
                    (int(np.sum(brainmask)),) + video_mats[1].shape, device=device
                )
                low_rank_video.set(torch.from_numpy(video_mats[0].T),
                                   torch.from_numpy(video_mats[1]))



                ##################################################
                ##################################################
                ##################################################
                if device=='cuda':
                    torch.cuda.synchronize()

                #
                print('v Rank Line Search')
                t0 = time.time()
                try:
                    locanmf_comps,loc_save = LocaNMF.rank_linesearch(low_rank_video,
                                                                 region_metadata,
                                                                 region_videos,
                                                                 maxiter_rank=maxrank,
                                                                 maxiter_lambda=maxiter_lambda,      # main param to tweak
                                                                 maxiter_hals=maxiter_hals,
                                                                 lambda_step=lambda_step,
                                                                 lambda_init=lambda_init,
                                                                 loc_thresh=loc_thresh,
                                                                 r2_thresh=r2_thresh,
                                                                 rank_range=rank_range,
                    #                                             nnt=nonnegative_temporal,
                                                                 verbose=[True, False, False],
                                                                 sample_prop=(1,1),
                                                                 device=device
                                                                )
                except:
                    print (" locaNMF Failed, skipping")
                    continue
                #
                if device=='cuda':
                    torch.cuda.synchronize()


                # C is the temporal components
                C = np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T
                print ("n_comps, n_time pts x n_trials: ", C.shape)
                qc, rc = np.linalg.qr(C.T)


                # Assigning regions to components
                region_ranks = []; region_idx = []

                for rdx in torch.unique(locanmf_comps.regions.data, sorted=True):
                    region_ranks.append(torch.sum(rdx == locanmf_comps.regions.data).item())
                    region_idx.append(rdx.item())

                areas=region_metadata.labels.data[locanmf_comps.regions.data].cpu().numpy()

                # Get LocaNMF spatial and temporal components
                A=locanmf_comps.spatial.data.cpu().numpy().T
                A_reshape=np.zeros((brainmask.shape[0],brainmask.shape[1],A.shape[1]));
                A_reshape.fill(np.nan)
                A_reshape[brainmask,:]=A

                # C is already computed above delete above
                if nonnegative_temporal:
                    C=locanmf_comps.temporal.data.cpu().numpy()
                else:
                    C=np.matmul(q,locanmf_comps.temporal.data.cpu().numpy().T).T

                # Add back removed columns from C as nans
                C_reshape=np.full((C.shape[0],dimsV[1]),np.nan)
                C_reshape[:,keepinds]=C
                C_reshape=np.reshape(C_reshape,[C.shape[0],dimsV[1]])

                # Get lambdas
                lambdas=np.squeeze(locanmf_comps.lambdas.data.cpu().numpy())


                # c_p is the trial sturcutre
                c_p=C_reshape.reshape(A_reshape.shape[2],int(C_reshape.shape[1]/1801),1801)

                #
                c_plot=c_p.transpose((1,0,2))
                c_plot.shape


                ##################################################
                ##################################################
                ##################################################
                # save LocaNMF data
                areas_saved = []
                for area in areas:
                    idx = np.where(rois_ids==np.abs(area))[0]
                    temp_name = str(rois_name[idx].squeeze())
                    if area <0:
                        temp_name += " - right"
                    else:
                        temp_name += " - left"

                    areas_saved.append(temp_name)

                # GET AREA NAMES
                def parse_areanames_new(region_name,rois_name):
                    areainds=[]; areanames=[];
                    for i,area in enumerate(region_name):
                        areainds.append(area)
                        areanames.append(rois_name[np.where(rois_ids==np.abs(area))][0])
                    sortvec=np.argsort(np.abs(areainds))
                    areanames=[areanames[i] for i in sortvec]
                    areainds=[areainds[i] for i in sortvec]
                    return areainds,areanames

                #
                region_name=region_mats[2]

                # Get area names for all components
                areainds,areanames_all = parse_areanames_new(region_name,rois_name)
                areanames_area=[]
                for i,area in enumerate(areas):
                    areanames_area.append(areanames_all[areainds.index(area)])

                ###################################
                np.savez(fname_out,
                          temporal_trial = c_plot[:int(c_plot.shape[0]/2),:,:],
                          temporal_random = c_plot[int(c_plot.shape[0]/2):,:,:],
                          areas = areas,
                          names = areas_saved,
                          A_reshape = A_reshape,
                          areanames_area = areanames_area
                         )






    def show_ROIs(self):

        session = self.sessions[0]

        fname_in = os.path.join(self.root_dir,self.animal_id,'tif_files',
                                  session,session+'_locanmf.npz')
        data = np.load(fname_in, allow_pickle=True)

        A_reshape = data["A_reshape"]
        areanames_area = data['areanames_area']

        ######################################################
        fig=plt.figure()
        for i in range(A_reshape.shape[2]):
            plt.subplot(4,4,i+1)
            plt.imshow(A_reshape[:,:,i])
            plt.title(areanames_area[i],fontsize=6)
        plt.tight_layout(h_pad=0.5,w_pad=0.5)
        plt.show()
