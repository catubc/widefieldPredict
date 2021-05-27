import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pickle as pk

import string, re
from scipy.signal import butter, filtfilt, cheby1
from sklearn.decomposition import PCA
from tqdm import tqdm
import parmap
from tqdm import tqdm, trange

import os

class ProcessCalcium():

    def __init__(self):

        pass

    def sum_pixels_in_registered_mask(self, data, maskwarp):

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


    def fix_trials(self, trial_courses, trial_courses_random):
        trial_courses_fixed = trial_courses.reshape(trial_courses.shape[0],-1)
        trial_courses_fixed = np.nan_to_num(trial_courses_fixed, nan=9999, posinf=9999, neginf=9999)
        idx = np.where(trial_courses_fixed==9999)

        trial_courses_fixed[idx]=0

        #print ('trial_courses_fixed: ', trial_courses_fixed.shape)
        trial_courses_random_fixed = trial_courses_random.copy().reshape(trial_courses_random.shape[0],-1)
        #print ('trial_courses_random_fixed: ', trial_courses_random_fixed.shape)
        trial_courses_random_fixed[:,idx[1]] = 0

        #
        trial_courses_fixed = trial_courses_fixed.reshape(trial_courses.shape)


        #
        trial_courses_random_fixed = trial_courses_random_fixed.reshape(trial_courses_random.shape)

        return trial_courses_fixed, trial_courses_random_fixed



    def get_04_triggers_with_lockout(self, root_dir, recording, lockout_window):

        # make sure locs
        try:
            locs_44threshold = np.load(root_dir + '/tif_files/' + recording + '/' + recording + '_locs44threshold.npy')
        except:
            print ("locs 44 thrshold missing", recording)
            return np.zeros((0),'float32'), np.zeros((0),'float32')

        #print ("Locs 44 threshold: ", locs_44threshold)
        codes = np.load(root_dir + '/tif_files/' + recording + '/'+recording + '_code44threshold.npy')
        code = b'04'
        idx = np.where(codes==code)[0]
        locs_selected = locs_44threshold[idx]

        if locs_selected.shape[0]==0:
            return np.zeros((0),'float32'), np.zeros((0),'float32')

        diffs = locs_selected[1:]-locs_selected[:-1]
        idx = np.where(diffs>lockout_window)[0]

        locs_selected_with_lockout = locs_selected[idx+1]
        if locs_selected_with_lockout.shape[0]==0:
            return np.zeros((0),'float32'), np.zeros((0),'float32')

        # ADD FIRST VAL
        if locs_selected[0]>lockout_window:
            locs_selected_with_lockout = np.concatenate(([locs_selected[0]], locs_selected_with_lockout), axis=0)

        # save data
        np.savetxt(root_dir + '/tif_files/' + recording + '/'+recording+ "_all_locs_selected.txt" ,
                   locs_selected)
        np.savetxt(root_dir + '/tif_files/' + recording + '/'+recording+ "_lockout_"+str(lockout_window)+
                   "sec_locs_selected.txt" , locs_selected_with_lockout)

        return locs_selected, locs_selected_with_lockout


    def find_nearest(self, array, value):
        return (np.abs(array-value)).argmin()

    def load_reclength(self, filename):
        """ Load realtime length of a single session. Probably should be in session, but was quicker to dump here"""

        #print ("FILENAME: ", filename)
        text_file = open(filename, "r")
        lines = text_file.read().splitlines()
        event_text = []
        for line in lines:
            event_text.append(re.split(r'\t+',line))

        #Delete false starts from event file
        for k in range(len(event_text)-1,-1,-1):        #Search backwards for the 1st occurence of "date" indicating last imaging start
                                                        #NB: There can be multiple false starts WHICH DON"T LINE UP - NEED TO IGNORE SUCH SESSIONS
            if event_text[k][0]=='date':
                event_text = event_text[k+2:]         #Remove first 2 lines
                break

        if len(event_text)==0:
            reclength = 0
        else:
            if event_text[-1][2] != "None":
                reclength = 0
            else:
                reclength = float(event_text[-1][3])

        return reclength


    # FUNCTION TO COMPUTE DFF
    def compute_DFF_function(self,
                            root_dir,
                            dff_method, # 'globalAverage' or 'slidingWindow'
                            recording,
                            locs_selected,
                            n_sec_window
                            ):


        # ###################################################
        # ###################################################
        # ###################################################
        # SET DEFAULT PARAMETERS
        #n_sec_window = 10
        low_cut = 0.1
        high_cut = 6.0
        img_rate = np.loadtxt(root_dir+'/img_rate.txt')
        selected_dff_filter = 'butterworth'

        # MAKE FILENAMES
        tif_files = root_dir+'/tif_files.npy'
        event_files = root_dir + '/event_files.npy'
        aligned_fname = root_dir + '/tif_files/'+recording + '/'+recording + "_aligned.npy"
        rec_filename = root_dir + '/tif_files/'+recording + '/'+recording +'.tif'
        n_sec = float(n_sec_window)

        # Load aligned/filtered data and find ON/OFF light;
        #images_file = self.parent.animal.home_dir+self.parent.animal.name+'/tif_files/'+self.rec_filename+'/'+self.rec_filename+'_aligned.npy'
        images_file = aligned_fname
        try:
            aligned_images = np.load(images_file)
        except:
            print ("missing aligned images - skipping session", recording)
            return np.zeros((0),'float32')


        # Find blue light on/off
        blue_light_threshold = 400  #Intensity threshold; when this value is reached - imaging light was turned on
        start_blue = 0; end_blue = len(aligned_images)

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

        #
        filtered_filename = images_file[:-4]+'_'+selected_dff_filter+'_'+str(low_cut)+'hz_'+str(high_cut)+'hz.npy'
        if os.path.exists(filtered_filename):
            try:
                aligned_images = np.load(filtered_filename, allow_pickle=True)
            except:
                print ("aligned filtered images corrupt... recomputing: ", filtered_filename)
                self.filter_data(root_dir, recording)
                aligned_images = np.load(filtered_filename)
        else:
            print ("aligned filtered images missing... recomputing: ", filtered_filename)
            self.filter_data(root_dir, recording)
            aligned_images = np.load(filtered_filename)

        aligned_images = aligned_images[start_blue:end_blue]

        #
        n_images=len(aligned_images)

        # Determine if imaging rate correct
        temp_tif_files = np.load(tif_files)
        temp_event_files = np.load(event_files)
        if len(temp_event_files)==1:
            temp_event_files = temp_event_files[0]

        if '4TBSSD' in self.main_dir:
            suffix = '4TBSSD'
        elif '1TB' in self.main_dir:
            suffix = '1TB'
        else:
            print ("New computer need to reset file locations")
            return None

        index = None
        for k in range(len(temp_tif_files)):
            try:
                temp_temp = temp_tif_files[k].decode("utf-8").replace('12TB/in_vivo/tim',suffix).replace(
                                                        '10TB/in_vivo/tim',suffix)#.replace("b'/", "'/")
            except:
                temp_temp = temp_tif_files[k].replace('12TB/in_vivo/tim',suffix).replace(
                                                        '10TB/in_vivo/tim',suffix)#.replace("b'/", "'/")
            if rec_filename in temp_temp:
                index = k
                break

        if index == None:
            print ("DID NOT FIND MATCH between imaging and lever ---- RETURNING ")

            # zero out locs selected because session can't be used
            fname_out1 = os.path.split(self.fname_04)[0]
            fname_out2 = os.path.split(fname_out1)[1]
            np.savetxt(self.fname_04[:-4]+"_all_locs_selected.txt" , [])
            np.savetxt(fname_out1+'/'+fname_out2+"_all_locs_selected.txt" , [])

            #print (temp_tif_files)
            return np.zeros((0),'float32')

        try:
            reclength = self.load_reclength(str(temp_event_files[index]).replace("b'",'').replace(
                                                "'",'').replace('12TB/in_vivo/tim',suffix))
        except:
            reclength = self.load_reclength(str(temp_event_files[index]).replace("b'",'').replace(
                                                "'",'').replace('10TB/in_vivo/tim',suffix))

        if reclength ==0:
            print ("zero length recording exiting (excitation light failure)", recording)
            # zero out locs selected because session can't be used
            fname_out1 = os.path.split(self.fname_04)[0]
            fname_out2 = os.path.split(fname_out1)[1]
            np.savetxt(self.fname_04[:-4]+"_all_locs_selected.txt" , [])
            np.savetxt(fname_out1+'/'+fname_out2+"_all_locs_selected.txt" , [])

            return np.zeros((0),'float32')

        # compute imaging rate;
        session_img_rate = n_images/reclength

        if abs(session_img_rate-float(img_rate))<0.01:         #Compare computed session img_rate w. experimentally set img_rate
            np.save(images_file.replace('_aligned.npy','')+'_img_rate', session_img_rate)
        else:
            np.save(images_file.replace('_aligned.npy','')+'_img_rate', session_img_rate)
            print ("Imaging rates between aligned and session are incorrect, exiting: ", session_img_rate)

            # zero out locs selected because session can't be used
            fname_out1 = os.path.split(self.fname_04)[0]
            fname_out2 = os.path.split(fname_out1)[1]
            np.savetxt(self.fname_04[:-4]+"_all_locs_selected.txt" , [])
            np.savetxt(fname_out1+'/'+fname_out2+"_all_locs_selected.txt" , [])

            return np.zeros((0),'float32')

        #
        trigger_times = locs_selected
        frame_times = np.linspace(0, reclength, n_images)             #Divide up reclength in number of images
        img_frame_triggers = []
        for i in range(len(trigger_times)):
            #img_frame_triggers.append(self.find_previous(frame_times, trigger_times[i]))
            img_frame_triggers.append(self.find_nearest(frame_times, trigger_times[i]))     #Two different functions possible here;

        #
        mean_file = root_dir + '/tif_files/'+recording + '/'+recording + '_aligned_mean.npy'
        if os.path.exists(mean_file)==False:
            aligned_fname = root_dir + '/tif_files/'+recording + '/'+recording + "_aligned.npy"
            images_file = aligned_fname
            images_aligned = np.load(images_file)
            images_aligned_mean = np.mean(images_aligned, axis=0)
            np.save(images_file[:-4]+'_mean', images_aligned_mean)

        global_mean = np.load(mean_file)

        abstimes = np.load(root_dir + '/tif_files/'+recording + '/'+recording + '_abstimes.npy')
        abspositions = np.load(root_dir + '/tif_files/'+recording + '/'+recording + '_abspositions.npy')

        data_stm = []; traces = []; locs = []; codes = []
        # counter=-1
        window = n_sec * session_img_rate      #THIS MAY NOT BE GOOD ENOUGH; SHOULD ALWAYS GO BACK AT LEAST X SECONDS EVEN IF WINDOW IS ONLY 1SEC or 0.5sec...
                                                                #Alternatively: always compute using at least 3sec window, and then just zoom in
        ##################################################
        ##################################################
        ##################################################
        data_stm = np.zeros((len(img_frame_triggers),(int(window)*2+1), 128, 128))
        counter = 0
        for trigger in img_frame_triggers:

            # NOTE: STARTS AND ENDS OF RECORDINGS MAY NOT HAVE PROPER [Ca] DATA; MAY NEED TO SKIP MANUALLY

            # load data chunk; make sure it is the right size; otherwise skip
            data_chunk = aligned_images[int(trigger-window):int(trigger+window)]
            #print (data_chunk.shape[0], window*2+1)
            if data_chunk.shape[0] != (int(window)*2+1):
                continue

            if dff_method == 'globalAverage':
                #data_stm.append(   #Only need to divide by global mean as original data_chunk did not have mean img added in
                temp = (data_chunk-global_mean)/global_mean
                data_stm[counter] = temp

            elif dff_method == 'slidingWindow':            #Use baseline -2*window .. -window
                print (" SLDING WINDOW METHOD NOT USED ANYMORE")
                return None
                if trigger < (2*window) or trigger>(n_images-window):
                    continue  #Skip if too close to start/end
                baseline = np.average(aligned_images[int(trigger-2*window):int(trigger-window)], axis=0)
                data_stm.append((data_chunk-baseline)/baseline)


            # advance the counter
            counter+=1

            # NOT USED ANYMORE
            # #***PROCESS TRACES - WORKING IN DIFFERENT TIME SCALE
            # lever_window = int(120*n_sec)    #NB: Lever window is computing in real time steps @ ~120Hz; and discontinuous;
            # t = np.linspace(-lever_window*0.0082,
            #                 lever_window*0.0082,
            #                 lever_window*2)
            # #
            # lever_position_index = self.find_nearest(np.array(abstimes), locs_selected[counter])
            # lever_trace = abspositions[int(lever_position_index-lever_window):int(lever_position_index+lever_window)]
            #
            # if len(lever_trace)!=len(t):    #Extraplote missing data
            #     lever_trace = np.zeros(lever_window*2,dtype=np.float32)
            #     for k in range(-lever_window,lever_window,1):
            #         lever_trace[k+lever_window] = self.abspositions[k+lever_window]     #Double check this...
            #
            # traces.append(lever_trace)

        # data_stm = np.array(data_stm)
        data_stm = data_stm[:counter]
        return data_stm


    def filter_data(self,
                    root_dir,
                    recording,
                    ):


        # ###################################################
        # ###################################################
        # ###################################################
        # SET DEFAULT PARAMETERS
        #n_sec_window = 10
        low_cut = 0.1
        high_cut = 6.0
        img_rate = 30.0
        selected_dff_filter = 'butterworth'

        # MAKE FILENAMES
        generic_mask_fname = root_dir + '/genericmask.txt'
        tif_files = root_dir+'tif_files.npy'
        event_files = root_dir + 'event_files.npy'
        aligned_fname = root_dir + '/tif_files/'+recording + '/'+recording + "_aligned.npy"

        #print ("FILTERING DATA: ", aligned_fname)

        # FILTERING STEP
        images_file = aligned_fname

        filter_type = selected_dff_filter
        lowcut = low_cut
        highcut = high_cut
        fs = img_rate

        # #Check to see if data requested exists- THIS CHECK WAS ALREADY DONE PRIOR TO ENTERING FUNCTION
        # if False:
        #     if os.path.exists(images_file[:-4]+'_'+filter_type+'_'+str(lowcut)+'hz_'+str(highcut)+'hz.npy'):
        #         #print ("filtered data already exists...")
        #         return

        #Load aligned images
        if os.path.exists(images_file):
            images_aligned = np.load(images_file)
        else:
            print (" ...missing aligned images... NEED TO RUN ALIGN ALGORITHMS", images_file)
            return None

            # TODO IMPLMENET ALIGNMENT TOOL
            #images_aligned = align_images2(self)

        #Save mean of images_aligned if not already done
        if os.path.exists(images_file[:-4]+'_mean.npy')==False:
            images_aligned_mean = np.mean(images_aligned, axis=0)
            np.save(images_file[:-4]+'_mean', images_aligned_mean)
        else:
            images_aligned_mean = np.load(images_file[:-4]+'_mean.npy')

        #Load mask - filter only datapoints inside mask
        n_pixels = len(images_aligned[0])
        generic_coords = np.loadtxt(generic_mask_fname)
        generic_mask_indexes=np.zeros((n_pixels,n_pixels))
        for i in range(len(generic_coords)): generic_mask_indexes[int(generic_coords[i][0])][int(generic_coords[i][1])] = True

        #Filter selection and parameters
        if filter_type == 'butterworth':
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            order = 2
            b, a = butter(order, [low, high], btype='band')
        elif filter_type == 'chebyshev':
            nyq = fs / 2.0
            order = 4
            rp = 0.1
            Wn = [lowcut / nyq, highcut / nyq]
            b, a = cheby1(order, rp, Wn, 'bandpass', analog=False)


        #Load individual pixel time courses; SWITCH TO UNRAVEL HERE****
        import time

        filtered_array = np.zeros(images_aligned.shape, dtype=np.float16)
        now = time.time(); start_time = now
        cutoff=n_pixels
        #from tqdm import tqdm
        #for p1 in tqdm(range(n_pixels)):
        for p1 in range(n_pixels):
            now=time.time(); n_pixels_in=0
            for p2 in range(n_pixels):
                if generic_mask_indexes[p1,p2]==False:
                    filtered_array[:,p1,p2] = np.float16(filtfilt(b, a, images_aligned[:,p1,p2])); n_pixels_in+=1   #filter pixel inside mask

        np.save(images_file[:-4]+'_'+filter_type+'_'+str(lowcut)+'hz_'+str(highcut)+'hz',
                filtered_array+np.float16(images_aligned_mean))

        return

    #select code 04/02/07 triggers;
    def get_triggers_bodyparts_whole_stack(self, recording):

        # find filename
        fname = os.path.join(self.main_dir,self.animal_id,'tif_files',recording,
                             recording+"_"+str(self.feature_quiescence)+"secNoMove_movements.npz")
        #print ("FNAME: movement feature: ", fname)
        try:
            data = np.load(fname, allow_pickle=True)
        except:
            print ("No video available: ", recording)
            return [], []

        #
        labels = data['labels']
        for k in range(len(labels)):
            if self.feature_name==labels[k]:
                feature_id = k
                break

        #
        feat = data['feature_movements']
        f = []
        for k in range(feat.shape[0]):
            temp = np.array(feat[k])#.T
            if temp.shape[0]>0:
                f.append(temp)

        feature_movements = np.vstack(f)

        # subsample these
        idx = np.random.choice(np.arange(feature_movements.shape[0]),min(100,feature_movements.shape[0]))
        feature_movements = feature_movements[idx]

        if feature_movements.shape[0]<=1:
            feature_starts = np.zeros((0),'float32')
        else:
            feature_starts = feature_movements[:,1]

        return feature_starts, []

    def get_triggers_bodyparts(self, recording):
        # find filename
        fname = os.path.join(self.main_dir,self.animal_id,'tif_files',recording,
                             recording+"_"+str(self.feature_quiescence)+"secNoMove_movements.npz")
        #print ("FNAME: movement feature: ", fname)
        try:
            data = np.load(fname, allow_pickle=True)
        except:
            print ("No video available: ", recording)
            return [], []

        #
        labels = data['labels']
        for k in range(len(labels)):
            if self.feature_name==labels[k]:
                feature_id = k
                break

        #
        feature_movements = np.array(data['feature_movements'][feature_id])
        #print (fname, feature_id, feature_movements)
        if feature_movements.shape[0]<=1:
            feature_starts = np.zeros((0),'float32')
        else:
            feature_starts = feature_movements[:,1]
        #print ("feature nomovements starts/ends: ", feature_movements)

        #if self.remove_shift:

       # print ("feature starts: ", feature_starts)

        return feature_starts, []

    #
    def compute_trial_courses_ROI_code04_trigger(self,
                                                 recording,
                                                 root_dir,
                                                 feature_name,
                                                 lockout_window,
                                                 n_sec_window,
                                                 recompute,
                                                 midline_filter_flag,
                                                 save_stm_flag,
                                                 transform_data_flag,
                                                 use_fixed_filter_flag,
                                                 fname_filter,
                                                 pca_denoise_flag
                                                 ):   # THIS IS THE DFF TIEM COURSE WINDOW; e.g. -10..+10sec


        # SET PARAMETERS
        # n_sec_window = 10
        dff_method = 'globalAverage'

        #
        if self.whole_stack == True:
            locs_selected, locs_selected_with_lockout = self.get_triggers_bodyparts_whole_stack(recording)

        else:
            print (" This notebook only used for whole stack, exiting")
            return

        #
        if len(locs_selected)==0:
            print (" ... session has no lever pulls ", recording)
            return
        
        # GENERATE SAVE FILENAMES FOR ALL CODE_04 DATA
        fname_04 = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        fname_random = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                        "_random_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        # good idea to save these as text to see them after:
        np.savetxt(fname_04[:-4]+"_locs_selected.txt" , locs_selected)

        #
        self.generate_arrays_ROI_triggered(root_dir,
                                             dff_method,
                                             recording,
                                             locs_selected,
                                             n_sec_window,
                                             fname_04,
                                             fname_random,
                                             recompute,
                                             midline_filter_flag,
                                             save_stm_flag,
                                             transform_data_flag,
                                             use_fixed_filter_flag,
                                             fname_filter,
                                             pca_denoise_flag
                                             )

        # just compute PCA and return;
        if self.feature_name=="whole_stack":
            return


        # GENERATE SAVE FILENAMES FOR LOCKOUT DATA
        fname_04 = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_lockout_"+str(lockout_window)+"sec_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")
        fname_random = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                        "_lockout_"+str(lockout_window)+"sec_random_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        #if os.path.exists(fname_04)==False:
        self.generate_arrays_ROI_triggered(root_dir,
                                             dff_method,
                                             recording,
                                             locs_selected_with_lockout,
                                             n_sec_window,
                                             fname_04,
                                             fname_random,
                                             recompute,
                                             midline_filter_flag,
                                             save_stm_flag,
                                             transform_data_flag,
                                             use_fixed_filter_flag,
                                             fname_filter,
                                         pca_denoise_flag)

    #
    def load_trial_courses_ROI_code04_trigger(self,
                                              recording,
                                              root_dir,
                                              feature_name,
                                              lockout_window,   # THIS IS THE LOCKOUT WINDOW FOR NO OTHER PULLS
                                              n_sec_window):   # THIS IS THE DFF TIEM COURSE WINDOW; e.g. -10..+10sec


        # GENERATE SAVE FILENAMES FOR ALL CODE_04 DATA
        fname_04 = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        fname_random = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                        "_random_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        data_04 = np.load(fname_04)
        data_04_random = np.load(fname_random)


        # GENERATE SAVE FILENAMES FOR LOCKOUT DATA
        fname_04 = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_lockout_"+str(lockout_window)+"sec_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")
        fname_random = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                        "_lockout_"+str(lockout_window)+"sec_random_ROItimeCourses_"+str(n_sec_window)+"sec.npy")


        data_04_lockout = np.load(fname_04)
        data_04_lockout_random = np.load(fname_random)


        return data_04, data_04_random, data_04_lockout, data_04_lockout_random


    def compute_midline_filter(self,
                               root_dir,
                               data_stm):
        import yaml
        print ("MIDLINE FILETER NOT USD")
        return None

        with open(os.path.join(root_dir,"gcamp.txt"), 'r') as f:
            valuesYaml = yaml.load(f, Loader=yaml.FullLoader)

        val999 = valuesYaml['val999']
        width = valuesYaml['width']
        power = valuesYaml['power']
        maxval = 10 # is not being used currently

        # generate midline filter using t=0 to approximately t=0.5sec brain activity which most likely to generate midline blood pressure
        # artifacts
        #print ("data stm: ", data_stm.shape)
        midline_filter = self.motion_mask_parallel(data_stm.mean(0)[data_stm.shape[0]//2:
                                                               data_stm.shape[0]//2+15].mean(0),
                                                              maxval,  # this value is not used
                                                              val999,
                                                              width,
                                                              power)

        return midline_filter


    def correct_midline_artifact(self,
                                 data_stm,
                                midline_filter,
                                transform):

        # scale the midline filter by the saved value from the transfomr file
        # loop over all frames in data stack and filter out midline activity
        for k in range(data_stm.shape[0]):
            for p in range(data_stm.shape[1]):
                data_stm[k,p] *=midline_filter**transform[4]

        return data_stm



    def rotate_translate_resize(self,
                                transform, data):

        from skimage.transform import resize, rotate

        #data = np.float64(data)

        # rotate
        data = rotate(data, transform[2])

        # roll
        data = np.roll(data, int(transform[0]), axis=1)
        data = np.roll(data, int(transform[1]), axis=0)

        # resize
        data = resize(data, (int(data.shape[0]*transform[3]),
                             int(data.shape[1]*transform[3])))

        # clip back down to 128 x 128
        data = data[data.shape[0]//2-64:data.shape[0]//2+64,
                    data.shape[0]//2-64:data.shape[0]//2+64]


        return data


    def transform_data(self,transform, stm):
        from tqdm import trange
        for k in trange(stm.shape[0]):
            for p in range(stm.shape[1]):
                stm[k,p] = self.rotate_translate_resize(transform, stm[k,p])

        return stm

    def generate_arrays_ROI_triggered(self,
                                      root_dir,
                                     dff_method,
                                     recording,
                                     locs_selected,
                                     n_sec_window,
                                     fname_04,
                                     fname_random,
                                     recompute,
                                     midline_filter_flag,
                                     save_stm_flag,
                                     transform_data_flag,
                                     use_fixed_filter_flag,
                                     fname_filter,
                                     pca_denoise_flag):

        fname_time_filters = fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy"
        fname_pca = fname_04[:-4]+"_pca.pkl"

        if os.path.exists(fname_pca):
            print ("  ... data already processed", fname_pca)
            return

        #
        self.fname_04 = fname_04
        from tqdm import trange
        if self.pca_etm:
            fname_out_final =  fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy"
            fname_out_random_final = fname_random[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy"

            if os.path.exists(fname_out_final) and os.path.exists(fname_out_random_final) and self.recompute==False:
                return

        else:
            if os.path.exists(fname_04) and os.path.exists(fname_random) and self.recompute==False:
                return

        # Compute DFF
        data_stm = self.compute_DFF_function(
                                root_dir,
                                dff_method, # 'globalAverage' or 'slidingWindow'
                                recording,
                                locs_selected,
                                n_sec_window
                                )

        ########################################################
        ########## COMPUTE DFF FOR TRIAL  DATA #################
        ########################################################
        # return if DFF data is none
        if data_stm.shape[0]==0:
            print ("data_stm is empty (could not compute stm, skipping : ", recording, " )")
            return

        # save data_stm stack
        if save_stm_flag:
            fname_04_data_stm = fname_04[:-4]+"_all_brain.npy"
            np.save(fname_04_data_stm, data_stm)

        # PCA denoising
        if self.pca_etm:

            fname_pca = fname_04[:-4]+"_pca.pkl"

            # USE EXISTING 10SEC PCA OBJECT; NO NEED DO RECOMPUTE IT
            if self.feature_name!='whole_stack':
                if os.path.exists(fname_pca)==False:
                    fname_pca = fname_pca.replace('15sec','10sec')

            if os.path.exists(fname_pca)==False:

                pca = self.pca_object(data_stm)

                # save the pca object for each dataset:
                # print (" ... saving pca: ", fname_pca)
                pk.dump(pca, open(fname_pca,"wb"))

            else:
                #print ("LOading pca: ", fname_pca)
                file = open(fname_pca, 'rb')

                #
                pca = pk.load(file)

            # compute PCA denoised STM for regular data;
            if True:
                pass
                #pca_etm_time_filters, pca_etm_spatial_filters = self.get_pca_filters_whole_stack(pca, data_stm)

                #np.save(fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy", pca_etm_time_filters)
                #np.save(fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+"_spatial.npy", pca_etm_spatial_filters)


            else:
                pca_etm_time_filters, pca_etm_spatial_filters = self.get_pca_filters(pca, data_stm)


                np.save(fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy", pca_etm_time_filters)
                np.save(fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+"_spatial.npy", pca_etm_spatial_filters)

        # compute ROI based etms
        else:
            # CONVERT DATA FROM 128 x 128 to 35 ROIs
            # load Allen Institute afine transformation to scale data
            if '4TBSSD' in self.main_dir:
                suffix = '4TBSSD'
            else:
                suffix = '1TB'
            maskwarp = np.load('/media/cat/'+suffix+'/yuki/maskwarp.npy')

            # accumulate mean activity in each ROI
            # input data shape: [# trials, # times, width, height]
            area_ids, trial_courses = self.sum_pixels_in_registered_mask(data_stm, maskwarp)

            # save area ids, time courses for event triggered and random data
            np.save(fname_04[:-4]+ "_area_ids.npy", area_ids)

        ########################################################
        ########## COMPUTE CONTROL/RANDOM DATA #################
        ########################################################
        # generate random time courses
        if self.feature_name!="whole_stack":

            locs_selected = get_random_times_outside_locs_selected(locs_selected,
                                                                   self.random_events_lockout,
                                                                   data_stm)

            np.savetxt(fname_04[:-4]+"_locs_selected_random.txt" , locs_selected)


            #locs_selected = np.float32(np.linspace(30, 1100, data_stm.shape[0]))
            #locs_selected = locs_selected + np.random.rand(locs_selected.shape[0])*10-5
            data_stm = None # zero out data_stm

            #
            # DFF for random data
            data_stm_random = self.compute_DFF_function(
                                    root_dir,
                                    dff_method, # 'globalAverage' or 'slidingWindow'
                                    recording,
                                    locs_selected,
                                    n_sec_window
                                    )

            if data_stm_random is None or data_stm_random.shape[0]==0:

                return

            # pca denoise
            if self.pca_etm:

                # use the PCA transform from above to denoise the random data also:
                pca_etm_time_filters, pca_etm_spatial_filters = self.get_pca_filters(pca, data_stm_random)
                np.save(fname_random[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy", pca_etm_time_filters)
                np.save(fname_random[:-4]+"_pca_"+str(self.pca_explained_var_val)+"_spatial.npy", pca_etm_spatial_filters)

            else:

                # compute random trial time courses
                _, trial_courses_random = self.sum_pixels_in_registered_mask(data_stm_random, maskwarp)
                data_stm_random = None

                #####################################################################
                ######## REMOVE INFINITIES, NANS ETC FROM DATA ######################
                #####################################################################
                if trial_courses.shape[0]==0 or trial_courses_random.shape[0]==0:
                    return np.zeros((0), 'float32'), np.zeros((0), 'float32')

                # remove infinities from both trial and randomized data
                trial_courses_fixed, trial_courses_random_fixed = self.fix_trials(trial_courses,
                                                                                  trial_courses_random)
                np.save(fname_random, trial_courses_random_fixed)
                np.save(fname_04, trial_courses_fixed)

        # return trial_courses_fixed, trial_courses_random_fixed


    def get_pca_filters_whole_stack(self, pca):

        # use set components for whole stack
        self.pca_fixed_comps = self.pca_n_components
        nComp = self.pca_fixed_comps

        print ("NOT IMPLENETED YET ")
        return
        # load entire calcium imaging data and compute dff
        fname_data = os.path.join(root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        #/media/cat/4TBSSD/yuki/IA1/tif_files/IA1pm_Feb1_30Hz/IA1pm_Feb1_30Hz_aligned_butterworth_0.1hz_6.0hz.npy
        print ("DATA_STM: ", data_stm.shape)
        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        #
        print("    denoising data (pca.transofrm(X)) ")
        time_filters = pca.transform(X)[:,:nComp]

        time_filters = np.array(time_filters)
        print("        done denoising data")
        pca_time_filters = time_filters.reshape(data_stm.shape[0],
                                                 data_stm.shape[1],
                                                 -1).transpose(0,2,1)
        pca_spatial_filters = pca.components_[:nComp,:]

        return pca_time_filters, pca_spatial_filters

    def get_pca_filters(self, pca, data_stm):

        # compute # of components needed for reconsturction to the requierd limit
        if self.pca_fixed_comps is None:
            expl_variance = pca.explained_variance_
            expl_variance = expl_variance/expl_variance.sum(0)
            sums = 0
            for k in range(expl_variance.shape[0]):
                sums+=expl_variance[k]
                if sums>=self.pca_explained_var_val:
                    nComp = k+1
                    break
        else:
            # just select a fixed number of comps
            nComp = self.pca_fixed_comps

        if self.pca_n_components != 0:
            self.pca_fixed_comps = self.pca_n_components
            nComp = self.pca_fixed_comps

        print ("DATA_STM: ", data_stm.shape)
        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        #
        print("    denoising data (pca.transofrm(X)) ")
        time_filters = pca.transform(X)[:,:nComp]

        time_filters = np.array(time_filters)
        print("        done denoising data")
        pca_time_filters = time_filters.reshape(data_stm.shape[0],
                                                 data_stm.shape[1],
                                                 -1).transpose(0,2,1)
        pca_spatial_filters = pca.components_[:nComp,:]

        return pca_time_filters, pca_spatial_filters

    def sigmoid_function(self, x, a, b):

        return np.clip(a*(np.ma.log(x) - np.ma.log(1 - x))+b, 0, 1)      #Compute sigmoid and cut off values below 0 and above 1


    def mangle(self, width, x, img_temp, maxval, power, val999):

        mu = 0 #Select approximate midline as centre of gaussian
        sig = width

        a = .005       #The steepness of the sigmoid function
        b = val999        #% of maxval to cutoff


        #Normalize img_temp for sigmoid to work properly
        #img_temp_norm = (img_temp-np.min(img_temp))/(np.max(img_temp) - np.min(img_temp))
        img_temp_norm = (img_temp-np.min(img_temp))/(np.max(img_temp) - np.min(img_temp))


        #Original root function
        #return -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * (abs(pix_val/maxval)**(1./power))
        return -np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) * self.sigmoid_function(img_temp_norm, a, b)


    # FUNCTION TO MASK
    def motion_mask_parallel(self, img_temp, maxval, val999, width, power):
        '''Parallel computation of mask
        '''

        y_array = []
        for x in range(len(img_temp)):
            y_array.append(np.arange(0,len(img_temp), 1))

        y_array = np.vstack(y_array)
        motion_mask = img_temp*self.mangle(width, np.abs(64-y_array), img_temp, maxval, power, val999)


        motion_mask = (motion_mask-np.min(motion_mask))/(np.max(motion_mask)-np.min(motion_mask))

        idx = np.where(motion_mask==0)
        motion_mask[idx]=1

        #motion_mask = motion_mask * img_temp

        return motion_mask

    def save_npz_data(self,
                      save_dir,
                      names,
                      lockout_window,
                      n_sec_window,
                      feature_name,
                      selected_sessions_animal,
                      best_sessions_animal,
                      save_best_flag):  #save only the best datasets

        #feature_name = 'code_04'
        #lockout_window = 10
        #n_sec_window = 10

        for name in names:

            #
            fname_out = save_dir+'/'+name+'.npz'

            #
            data_04_list = []
            data_04_random_list = []
            data_04_lockout_list = []
            data_04_lockout_random_list = []

            session_list = []
            ctr_list = []


            root_dir = '/media/cat/4TBSSD/yuki/'+name

            temp_recs = np.load(root_dir+'/tif_files.npy')
            recordings =[]
            for k in range(len(temp_recs)):
                try:
                    recordings.append(str(os.path.split(temp_recs[k])[1][:-4], "utf-8"))
                except:
                    recordings.append(os.path.split(temp_recs[k])[1][:-4])

            print ("PROCESSING: ", name)

            print ("rec id,      rec name,           all rewarded trials,   "+
                   str(n_sec_window) + " sec lockout rewarded trials (*** good sessions; ####### best 3 sessions")
            for ctr,recording in enumerate(recordings):

                # MAKE PRINTOUT TABLE

                if save_best_flag==False:
                    prefix = '       '
                    if ctr in selected_sessions_animal:
                        if ctr in best_sessions_animal:
                            prefix="#######"
                        else:
                            prefix='    ***'

                else:
                    if ctr in selected_sessions_animal:
                        if ctr in best_sessions_animal:
                            prefix="#######"
                        else:
                            prefix='    ***'
                    else:
                        continue

                try:
                    (data_04, data_04_random, data_04_lockout, data_04_lockout_random) = self.load_trial_courses_ROI_code04_trigger(
                                                                                                          recording,
                                                                                                          root_dir,
                                                                                                          feature_name,
                                                                                                          lockout_window,
                                                                                                          n_sec_window)
                    data_04_list.append(data_04)
                    data_04_random_list.append(data_04_random)
                    data_04_lockout_list.append(data_04_lockout)
                    data_04_lockout_random_list.append(data_04_lockout_random)

                    session_list.append(recording)
                    ctr_list.append(ctr)

                except:
                    data_04 = np.zeros((0),'float32')
                    data_04_random = data_04
                    data_04_lockout = data_04
                    data_04_lockout_random = data_04

                    data_04_list.append(data_04)
                    data_04_random_list.append(data_04)
                    data_04_lockout_list.append(data_04)
                    data_04_lockout_random_list.append(data_04)

                    session_list.append(recording)
                    ctr_list.append(ctr)



                print (prefix,ctr, "     ", recording,"    ", data_04.shape, "        ", data_04_lockout.shape)

            np.savez(fname_out,
                     data_04 = data_04_list,
                     data_04_random = data_04_random_list,
                     data_04_lockout = data_04_lockout_list,
                     data_04_lockout_random= data_04_lockout_random_list,
                     session_list = session_list,
                     ctr_list = ctr_list,
                     selected_sessions = selected_sessions_animal,
                     best_sessions = best_sessions_animal)

    def pca_object(self, data_stm):

        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        # subselect data
        n_selected = min(X.shape[0]//10, 10000)
        idx = np.random.choice(np.arange(X.shape[0]),
                               n_selected,
                               replace=False)
        X_select = X[idx]

        pca = PCA()
        pca.fit(X_select)

        return pca

    def pca_denoise(self, data_stm, nComp, pca):

        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        mu= np.mean(X, axis=0)

        Xnew = np.dot(pca.transform(X)[:,:nComp],   # time
                     pca.components_[:nComp,:])     # space

        Xnew+=mu

        data_stm_denoised_reshaped = Xnew.reshape(data_stm.shape[0],data_stm.shape[1],
                                                  data_stm.shape[2], data_stm.shape[3])

        return data_stm_denoised_reshaped


    def generate_etm(self,
                     names,
                     n_sec_window,
                     lockout_window,
                     recompute,
                     pca_etm,
                     pca_explained_var_val,
                     parallel=False,
                     n_cores=2):

        self.pca_etm = pca_etm

        self.pca_explained_var_val = pca_explained_var_val

        # midline filter params- TURN OFF FOR NOW
        midline_filter_flag = False
        use_fixed_filter_flag = False  # use filter from a single recording on all data
        fname_filter = '/media/cat/4TBSSD/yuki/IA1/tif_files/IA1am_Mar11_30Hz/IA1am_Mar11_30Hz_code_04_trial_ROItimeCourses_15sec_midline_filter.npy'

        #
        feature_name = self.feature_name
        save_stm_flag = False         # flag to save raw stm maps [128 x 128 x n_times] files during processing
        transform_data_flag = False   # flag which reverts to using manually aligned/transformed data

        #
        pca_denoise_flag = False

        for name in names:
            self.animal_id = name
            root_dir = self.main_dir + name

            temp_recs = np.load(root_dir + '/tif_files.npy')
            recordings =[]
            for k in range(len(temp_recs)):
                try:
                    recordings.append(str(os.path.split(temp_recs[k])[1][:-4], "utf-8"))
                except:
                    recordings.append(os.path.split(temp_recs[k])[1][:-4])

            print ("PROCESSING: ", name)

            if parallel:
                print ("TO IMPLMENET SELECTED SESSIONS IN PARALLEL")
                res = parmap.map(self.compute_trial_courses_ROI_code04_trigger,
                                   recordings,
                                   root_dir,
                                   feature_name,
                                   lockout_window,
                                   n_sec_window,
                                   recompute,
                                   midline_filter_flag,
                                   save_stm_flag,
                                   transform_data_flag,
                                   use_fixed_filter_flag,
                                   fname_filter,
                                   pca_denoise_flag,
                                   pm_processes=n_cores,
                                   pm_pbar=True)
            else:
                for recording in tqdm(recordings):

                    if self.skip_to != None:
                        if self.skip_to in recording:
                            self.skip_to = None
                        print ("   skipping: ", recording)
                        continue

                    if  self.sessions in recording or self.sessions=='all':

                        #
                        self.compute_trial_courses_ROI_code04_trigger(recording,
                                                               root_dir,
                                                               feature_name,
                                                               lockout_window,
                                                               n_sec_window,
                                                               recompute,
                                                               midline_filter_flag,
                                                               save_stm_flag,
                                                               transform_data_flag,
                                                               use_fixed_filter_flag,
                                                               fname_filter,
                                                               pca_denoise_flag)


    # def generate_etm_whole_stack(self,
    #                          names,
    #                          n_sec_window,
    #                          lockout_window,
    #                          recompute,
    #                          pca_etm,
    #                          pca_explained_var_val,
    #                          parallel=False,
    #                          n_cores=2):
    #
    #     self.pca_etm = pca_etm
    #
    #     self.pca_explained_var_val = pca_explained_var_val
    #
    #     # midline filter params- TURN OFF FOR NOW
    #     midline_filter_flag = False
    #     use_fixed_filter_flag = False  # use filter from a single recording on all data
    #     fname_filter = '/media/cat/4TBSSD/yuki/IA1/tif_files/IA1am_Mar11_30Hz/IA1am_Mar11_30Hz_code_04_trial_ROItimeCourses_15sec_midline_filter.npy'
    #
    #     #
    #     feature_name = self.feature_name
    #     save_stm_flag = False         # flag to save raw stm maps [128 x 128 x n_times] files during processing
    #     transform_data_flag = False   # flag which reverts to using manually aligned/transformed data
    #
    #     #
    #     pca_denoise_flag = False
    #
    #     for name in names:
    #         self.animal_id = name
    #         root_dir = self.main_dir + name
    #
    #         temp_recs = np.load(root_dir + '/tif_files.npy')
    #         recordings =[]
    #         for k in range(len(temp_recs)):
    #             try:
    #                 recordings.append(str(os.path.split(temp_recs[k])[1][:-4], "utf-8"))
    #             except:
    #                 recordings.append(os.path.split(temp_recs[k])[1][:-4])
    #
    #         print ("PROCESSING: ", name)
    #
    #         if parallel:
    #             print ("TO IMPLMENET SELECTED SESSIONS IN PARALLEL")
    #             res = parmap.map(self.compute_trial_courses_ROI_code04_trigger,
    #                                recordings,
    #                                root_dir,
    #                                feature_name,
    #                                lockout_window,
    #                                n_sec_window,
    #                                recompute,
    #                                midline_filter_flag,
    #                                save_stm_flag,
    #                                transform_data_flag,
    #                                use_fixed_filter_flag,
    #                                fname_filter,
    #                                pca_denoise_flag,
    #                                pm_processes=n_cores,
    #                                pm_pbar=True)
    #         else:
    #             for recording in tqdm(recordings):
    #
    #                 if self.skip_to != None:
    #                     if self.skip_to in recording:
    #                         self.skip_to = None
    #                     print ("   skipping: ", recording)
    #                     continue
    #
    #                 if  self.sessions in recording or self.sessions=='all':
    #
    #                     #
    #                     self.compute_trial_courses_ROI_code04_trigger(recording,
    #                                                            root_dir,
    #                                                            feature_name,
    #                                                            lockout_window,
    #                                                            n_sec_window,
    #                                                            recompute,
    #                                                            midline_filter_flag,
    #                                                            save_stm_flag,
    #                                                            transform_data_flag,
    #                                                            use_fixed_filter_flag,
    #                                                            fname_filter,
    #                                                            pca_denoise_flag)
    #


def get_random_times_outside_locs_selected(locs_selected,
                                           random_lockout,
                                           data_stm):
    #
    random_selected = []
    for k in range(10000):
        time = np.random.rand()*1200 + self.n_sec_window  # assumes data chunk is at least 1200 seconds

        if np.min(np.abs(time-locs_selected))>random_lockout:
            random_selected.append(time)

        if len(random_selected)==data_stm.shape[0]:
            # print ("Found sufficient randomized data chunks")
            break

    if k==10000:
        print ("Not enough random data chunkcs could not be generated with lockout window size: ",
               self.random_events_lockout)
        return None

    random_selected = np.sort(random_selected)
    return random_selected
