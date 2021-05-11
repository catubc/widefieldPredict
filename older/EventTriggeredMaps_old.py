import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pickle as pk

import string, re
from scipy.signal import butter, filtfilt, cheby1
from sklearn.decomposition import PCA
from tqdm import tqdm
import parmap

import os

class EventTriggeredMaps():

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

    #select code 04/02/07 triggers;
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
        #print ("aligned fname: ", aligned_fname)

        rec_filename = root_dir + '/tif_files/'+recording + '/'+recording +'.tif'
        #print ("rec_fileame;", rec_filename)
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


        #self.lowcut = float(self.parent.filter_low.text())
        #self.highcut = float(self.parent.filter_high.text())

        #if self.selected_dff_filter == 'nofilter':
        #    pass; #already loaded nonfiltered self.aligned_images above
        #else:
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

        # compute # of images in stack
        n_images=len(aligned_images)


        # Determine if imaging rate correct
        temp_tif_files = np.load(tif_files)
        temp_event_files = np.load(event_files)
        if len(temp_event_files)==1:
            temp_event_files = temp_event_files[0]
        #print ("temp_tif files;l", temp_tif_files)
        #print ("rec_filename: ", rec_filename)

        index = None
        if '4TBSSD' in self.main_dir:
            suffix = '4TBSSD'
        elif '1TB' in self.main_dir:
            suffix = '1TB'
        else:
            print ("New computer need to reset file locations")
            return None

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
            #print (temp_tif_files)
            return np.zeros((0),'float32')

        #print ("INDEX: ", index)
        #print ("temp event files indexed: ", len(temp_event_files[index]))
        # load the reclength based
        #print (" RECLEN FILE: ", temp_event_files[index])
        try:
            reclength = self.load_reclength(temp_event_files[index].replace(
                                                        '12TB/in_vivo/tim',suffix))
        except:
            reclength = self.load_reclength(temp_event_files[index].replace(
                                                        '10TB/in_vivo/tim',suffix))

        if reclength ==0:
            print ("zero length recording exiting (excitation light failure)", recording)
            return np.zeros((0),'float32')

        # compute imaging rate;
        session_img_rate = n_images/reclength

        if abs(session_img_rate-float(img_rate))<0.01:         #Compare computed session img_rate w. experimentally set img_rate
            np.save(images_file.replace('_aligned.npy','')+'_img_rate', session_img_rate)
        else:
            np.save(images_file.replace('_aligned.npy','')+'_img_rate', session_img_rate)
            print ("Imaging rates between aligned and session are incorrect, exiting: ", session_img_rate)
            return np.zeros((0),'float32')


        # Find times of triggers from lever pull threshold times
        trigger_times = locs_selected
        frame_times = np.linspace(0, reclength, n_images)             #Divide up reclength in number of images
        img_frame_triggers = []
        for i in range(len(trigger_times)):
            #img_frame_triggers.append(self.find_previous(frame_times, trigger_times[i]))
            img_frame_triggers.append(self.find_nearest(frame_times, trigger_times[i]))     #Two different functions possible here;

        #BASELINE FOR GLOBAL BASELINE REMOVAL
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
        counter=-1
        window = n_sec * session_img_rate      #THIS MAY NOT BE GOOD ENOUGH; SHOULD ALWAYS GO BACK AT LEAST X SECONDS EVEN IF WINDOW IS ONLY 1SEC or 0.5sec...
                                                                #Alternatively: always compute using at least 3sec window, and then just zoom in
        #print ("tirggers: ", img_frame_triggers)
        for trigger in img_frame_triggers:
            counter+=1
            #NB: Ensure enough space for the sliding window; usually 2 x #frames in window
            if trigger < (2*window) or trigger>(n_images-window):
                continue  #Skip if too close to start/end

            #add locs and codes
            #locs.append(locs_44threshold_selected[counter])
            #codes.append(code_44threshold_selected[counter])

            # load data chunk working with
            data_chunk = aligned_images[int(trigger-window):int(trigger+window)]

            if dff_method == 'globalAverage':
                data_stm.append((data_chunk-global_mean)/global_mean)    #Only need to divide by global mean as original data_chunk did not have mean img added in

            elif dff_method == 'slidingWindow':            #Use baseline -2*window .. -window
                baseline = np.average(aligned_images[int(trigger-2*window):int(trigger-window)], axis=0)
                data_stm.append((data_chunk-baseline)/baseline)

            #***PROCESS TRACES - WORKING IN DIFFERENT TIME SCALE
            lever_window = int(120*n_sec)    #NB: Lever window is computing in real time steps @ ~120Hz; and discontinuous;
            t = np.linspace(-lever_window*0.0082,
                            lever_window*0.0082,
                            lever_window*2)
            #lever_position_index = find_nearest(np.array(self.abstimes), self.locs_44threshold[counter])
            lever_position_index = self.find_nearest(np.array(abstimes), locs_selected[counter])

            lever_trace = abspositions[int(lever_position_index-lever_window):int(lever_position_index+lever_window)]

            if len(lever_trace)!=len(t):    #Extraplote missing data
                lever_trace = np.zeros(lever_window*2,dtype=np.float32)
                for k in range(-lever_window,lever_window,1):
                    lever_trace[k+lever_window] = self.abspositions[k+lever_window]     #Double check this...

            traces.append(lever_trace)

        data_stm = np.array(data_stm)

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

        #Check to see if data requested exists- THIS CHECK WAS ALREADY DONE PRIOR TO ENTERING FUNCTION
        if False:
            if os.path.exists(images_file[:-4]+'_'+filter_type+'_'+str(lowcut)+'hz_'+str(highcut)+'hz.npy'):
                #print ("filtered data already exists...")
                return

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

        locs_selected, locs_selected_with_lockout = self.get_04_triggers_with_lockout(root_dir,
                                                                                    recording,
                                                                                    lockout_window)
        if len(locs_selected)==0:
            print (" ... session does not have lever pulls ")
            return
        
        # GENERATE SAVE FILENAMES FOR ALL CODE_04 DATA
        fname_04 = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                    "_trial_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        fname_random = (root_dir + '/tif_files/' + recording + '/' + recording + "_"+feature_name+
                        "_random_ROItimeCourses_"+str(n_sec_window)+"sec.npy")

        # good idea to save these as text to see them after:
        np.savetxt(fname_04[:-4]+"_locs_selected.txt" , locs_selected)


        #if os.path.exists(fname_04)==False:
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


        #return dff1, dff1_random, dff2, dff2_random



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

        from tqdm import trange

        # Compute DFF
        data_stm = self.compute_DFF_function(
                                root_dir,
                                dff_method, # 'globalAverage' or 'slidingWindow'
                                recording,
                                locs_selected,
                                n_sec_window
                                )

        # return if DFF data is none
        if data_stm.shape[0]==0:
            print ("data_stm is empty (could not compute stm, skipping)", recording)
            return np.zeros((0), 'float32'), np.zeros((0), 'float32')

        # try to load transform coordinates to assess filter power
        try:
            transform = np.load(os.path.split(fname_04)[0]+"/transform.npy")
        except:
            # set dummy values so the filter power can be set to 1 below
            transform = [0,  # xtranslate
                         0,  # ytranslate
                         0,  # rotation
                         1,  # resizing
                         1]  # power of filter

        # Filter to remove midline artifacts etc.
        if midline_filter_flag:

            if use_fixed_filter_flag:
                midline_filter = np.load(fname_filter,allow_pickle='True')
                print ("using fixed filter")

                # must also force the same filter on all the data;
                transform = [0,  # xtranslate
                             0,  # ytranslate
                             0,  # rotation
                             1,  # resizing
                             1]  # power of filter

            else:
                midline_filter = self.compute_midline_filter(root_dir,
                                              data_stm)
                midline_filter.dump(fname_04[:-4]+ "_midline_filter.npy")

            data_stm = self.correct_midline_artifact(data_stm,
                                                midline_filter,
                                                transform)  # pass in filter power value to decrease the amount of filtering if necessary




        # check to see if data requires TRANSOFMRATIONS: rotations, etc.
        if transform_data_flag:
            try:
                transform = np.load(os.path.split(fname_04)[0]+"/transform.npy")
                data_stm = self.transform_data(transform, data_stm)
            except:
                print ("Transform coordinates missing ... skipping")

        # save data_stm stack
        if save_stm_flag:
            fname_04_data_stm = fname_04[:-4]+"_all_brain.npy"
            np.save(fname_04_data_stm, data_stm)


        # PCA denoising
        if self.pca_etm:

            fname_pca = fname_04[:-4]+"_pca.pkl"

            if os.path.exists(fname_pca)==False:

                pca = self.pca_object(data_stm)

                # save the pca object for each dataset:
                print ("fname pca: ", fname_pca)

                pk.dump(pca, open(fname_pca,"wb"))
                np.save(fname_pca[:-4]+"_explained_variance.npy", expl_variance)

            else:
                file = open(fname_pca, 'rb')

                # dump information to that file
                pca = pk.load(file)

                expl_variance = pca.explained_variance_
                expl_variance = expl_variance/expl_variance.sum(0)
                # print ("explained variance: ", expl_variance)

                # denoise data using pca object; also use it below

            # compute
            pca_etm = self.get_pca_filters(pca, data_stm)
            #np.save(fname_04[:-4]+ "_area_ids.npy", area_ids)
            np.save(fname_04[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy", pca_etm)

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



        ########################################################
        ########## COMPUTE CONTROL/RANDOM DATA #################
        ########################################################
        # generate random time corses
        locs_selected = np.float32(np.linspace(30, 1100, data_stm.shape[0]))
        locs_selected = locs_selected + np.random.rand(locs_selected.shape[0])*10-5
        data_stm = None


        # DFF for random data
        data_stm_random = self.compute_DFF_function(
                                root_dir,
                                dff_method, # 'globalAverage' or 'slidingWindow'
                                recording,
                                locs_selected,
                                n_sec_window
                                )

        if data_stm_random is None:
            return np.zeros((0), 'float32'), np.zeros((0), 'float32')

        # use the same filter as in the event triggered neural activity
        if midline_filter_flag:
            data_stm_random = self.correct_midline_artifact(data_stm_random,
                                                       midline_filter,
                                                       transform)

        # transform random data: rotations, etc.
        # check to see if data requires TRANSOFMRATIONS: rotations, etc.
        if transform_data_flag:
            try:
                transform = np.load(os.path.split(fname_04)[0]+"/transform.npy")
                data_stm_random = self.transform_data(transform, data_stm_random)
            except:
                print ("Transform coordinates missing ... skipping")


        # pca denoise
        if self.pca_etm:

            # use the PCA transform from above to denoise the random data also:
            # compute
            random_pca_etm = self.get_pca_filters(pca, data_stm_random)
            np.save(fname_random[:-4]+"_pca_"+str(self.pca_explained_var_val)+".npy", random_pca_etm)

        else:

            # compute random trial time courses
            _, trial_courses_random = self.sum_pixels_in_registered_mask(data_stm_random, maskwarp)
            data_stm_random = None

            #####################################################################
            ######## REMOVE INFINITIES, NANS ETC FROM DATA ######################
            #####################################################################
            if trial_courses.shape[0]==0 or trial_courses_random.shape[0]==0:
                return np.zeros((0), 'float32'), np.zeros((0), 'float32')

            # remove infiities
            trial_courses_fixed, trial_courses_random_fixed = self.fix_trials(trial_courses,
                                                                              trial_courses_random)
            # save area ids, time courses for event triggered and random data
            np.save(fname_04[:-4]+ "_area_ids.npy", area_ids)
            np.save(fname_04, trial_courses_fixed)
            np.save(fname_random, trial_courses_random_fixed)

        # return trial_courses_fixed, trial_courses_random_fixed

    def get_pca_filters(self, pca, data_stm):
        expl_variance = pca.explained_variance_
        expl_variance = expl_variance/expl_variance.sum(0)
        sums = 0
        for k in range(expl_variance.shape[0]):
            sums+=expl_variance[k]
            if sums>=self.pca_explained_var_val:
                nComp = k+1
                break

        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])
        time_filters = pca.transform(X)[:,:nComp]
        pca_time_filters_only = time_filters.reshape(data_stm.shape[0], data_stm.shape[1],-1).transpose(0,2,1)

        return pca_time_filters_only

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

    def pca_object(data_stm):

        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        # subselect data
        n_selected = min(1500, X.shape[0])
        idx = np.random.choice(np.arange(X.shape[0]),n_selected,
                               replace=False)
        X_select = X[idx]

        pca = PCA()
        pca.fit(X_select)

        return pca

    def pca_denoise(data_stm, nComp, pca):

        X = data_stm.reshape(data_stm.shape[0]*data_stm.shape[1],
                             data_stm.shape[2]*data_stm.shape[3])

        mu= np.mean(X, axis=0)

        Xnew = np.dot(pca.transform(X)[:,:nComp],
                     pca.components_[:nComp,:])

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
        feature_name = 'code_04'
        save_stm_flag = False         # flag to save raw stm maps [128 x 128 x n_times] files during processing
        transform_data_flag = False   # flag which reverts to using manually aligned/transformed data

        # denoise PCA
        pca_denoise_flag = False

        for name in names:
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

                    if  self.sessions in recording or self.sessions=='all':
                        print ("recording: ", recording)
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

        print ("DONE STMs....")
