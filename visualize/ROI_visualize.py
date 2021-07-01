import matplotlib

import matplotlib.pyplot as plt

import numpy as np
import os

import glob2


#
def sum_pixels_in_registered_mask(data,
                                  maskwarp,
                                  min_pixels=10):


    #
    areas = np.unique(maskwarp)



    # work in 1D vectors easier to mask
    maskwarp1D = maskwarp.reshape(-1)
    #maskwarp1D = maskwarp

    trial_courses = []
    area_ids = []
    for k in range(data.shape[0]):
        time_courses_local = []

        # convert to 1D vector to mask faster
        data1D = np.float32(data[k].reshape(data[k].shape[0],-1))

        #
        print (k,"Data1D: ", data1D.shape)
        for id_ in areas:
            idx = np.where(maskwarp1D==id_)[0]
            #idx = np.where(maskwarp1D==id_)

            # only keep areas that have at least 10 pixels
            if idx.shape[0]>min_pixels:
                area_ids.append(id_)

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
    print ("# trials, # areas, # times: ", trial_courses.shape)
    print ("area ids: ", area_ids.shape)

    return area_ids, trial_courses

#
def check_neighbours(mask_roi, idx, p):

    x = idx[0][p]
    y = idx[1][p]

    sums = 0
    for k in range(-1,2,1):
        for p in range(-1,2,1):
            xx = x+k
            yy = y+p
            if xx < 0 or xx> 127 or yy<0 or yy>127:
                pass
            else:
                sums+=mask_roi[xx,yy]
                if sums>=2:
                    return True
    return False

#
def remove_single_pixels(maskwarp,
                         root_dir,
                         min_pixels=10):



    # load brain mask (i.e. no tissue areas) and apply it to the image mask first
    temp = np.int32(np.loadtxt(root_dir+'genericmaskIJ2.txt'))
    brain_mask = np.ones((128,128),'float32')
    for t in temp:
        brain_mask[t[0],t[1]]=0

    maskwarp = maskwarp*brain_mask

    #
    ids = np.unique(maskwarp)

    ids_selected = []
    masks = []
    mask_b = np.zeros((128,128),'int32')
    for id_ in ids:
        idx = np.where(maskwarp==id_)
        if idx[0].shape[0]>min_pixels:

            # make the mask for the specific ROI
            mask_roi = np.zeros((128,128))
            idx = np.where(maskwarp==id_)

            mask_roi[idx] = 1

            #
            x = []
            y = []
            for p in range(idx[0].shape[0]):
                connected = check_neighbours(mask_roi, idx, p)
                if connected:
                    x.append(idx[0][p])
                    y.append(idx[1][p])

            #
            if len(x)>min_pixels:
                ids_selected.append(int(id_))
                temp = mask_b.copy()
                for p in range(len(x)):
                     temp[x[p],y[p]]=1

                masks.append(temp)


    ids_selected = np.array(ids_selected)
    return ids_selected, masks

def plot_rois(ids, masks, mask,names, min_pixels):
    # Plot ROIs
    rois = []
    final_names = []
    for ctr, id_ in enumerate(ids):
        ax=plt.subplot(5,7,ctr+1)
        temp = np.zeros((128,128))+np.nan

        roi = masks[ctr]*mask
        plt.imshow(roi, interpolation=None)
        rois.append(roi)
        #
        plt.xticks([])
        plt.yticks([])
        plt.ylabel(str(id_))
        plt.title(names[id_], fontsize=6)
        final_names.append(names[id_])

    plt.suptitle(str(masks.shape[0]) +" ROIS w. "+str(min_pixels)+" minimum pixels inside widefield mask")
    plt.show()

    return rois


# Load names of ROIs
def load_mask_and_names(root_dir):

    with open(root_dir+'dorsalMaps_name.txt') as f:
        fs = list(f)

    fs = np.vstack(fs)
    names = []
    for name in fs:
        temp = name[0][1:-2]
        names.append(temp)

    # load brain mask (i.e. no tissue areas)
    temp = np.int32(np.loadtxt(root_dir+'genericmaskIJ2.txt'))
    mask = np.ones((128,128),'float32')
    for t in temp:
        mask[t[0],t[1]]=np.nan

    return mask, names



def cleanup_rois(root_dir, min_pixels = 50):
    maskwarp= np.load(root_dir+'maskwarp.npy')

    # Auto clean up areas
    ids, masks = remove_single_pixels(maskwarp,
                                      root_dir,
                                      min_pixels)

    # Manually remove specific areas:
    idx_del =  [0,16,20]
    ids = np.delete(ids,0,0)
    masks = np.delete(masks,0,0)

    print ("FINAL ROIS: ", ids, len(ids))

    return masks, ids
