#Function that computes power spetrum density (PSD) and specgrams for ephys data. 
# 

import struct
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

def Specgram(data, sampfreq, p0=-40, f0 = 0.1, f1 = 110,
             width = 2, tres=0.5):

    t0=None
    t1=None
    #f0=0.1
    #f1=110
    #p0=-60
    p1=None
    #chanis=-1
    #width=2
    #tres=.5
    #cm=None
    #colorbar=False
    #title=True
    #figsize=(20, 6.5)
    
    P0, P1 = None, None

    #sampfreq=SampleFrequency #1KHZ LFP SAMPLE RATE for Nick's data; Otherwise full sample rates;

    #NFFT = intround(width * sampfreq)
    #NOVERLAP = intround(NFFT - tres * SAMPFREQ)

    length = len(data)

    ts = np.arange(0,len(data),1.0)/sampfreq

    if t0 == None:
        t0, t1 = ts[0], ts[-1] # full duration
    #if t1 == None:
    #    t1 = t0 + 10 # 10 sec window
    if width == None:
        width = uns['LFPWIDTH'] # sec
    if tres == None:
        tres = uns['LFPTRES'] # sec
    assert tres <= width

    NFFT = intround(width * sampfreq)
    noverlap = intround(NFFT - tres * sampfreq)

    t0i, t1i = ts.searchsorted((t0, t1))

    #data = filter.notch(data)[0] # remove 60 Hz mains noise

    #print ("Computing regular fft specgram")
    P, freqs, t = mpl.mlab.specgram(data/1e3, NFFT=NFFT, Fs=sampfreq, noverlap=noverlap)
    
    # convert t to time from start of acquisition:
    t += t0
    # keep only freqs between f0 and f1:
    if f0 == None:
        f0 = freqs[0]
    if f1 == None:
        f1 = freqs[-1]
    lo, hi = freqs.searchsorted([f0, f1])
    P, freqs = P[lo:hi], freqs[lo:hi]
    #print P
    
    # check for and replace zero power values (ostensibly due to gaps in recording)
    # before attempting to convert to dB:
    zis = np.where(P == 0.0) # row and column indices where P has zero power
    if len(zis[0]) > 0: # at least one hit
        P[zis] = np.finfo(np.float64).max # temporarily replace zeros with max float  #CAT: This can probably be unhacked using nanmax or masked arrays
        minnzval = P.min() # get minimum nonzero value
        P[zis] = minnzval # replace with min nonzero values
    P = 10. * np.log10(P) # convert power to dB wrt 1 mV^2?

    # for better visualization, clip power values to within (p0, p1) dB
    if p0 != None:
        P[P < p0] = p0
    if p1 != None:
        P[P > p1] = p1

    extent = ts[0], ts[-1], freqs[0], freqs[-1]

    return P[::-1], extent

  
def intround( n):
    """Round to the nearest integer, return an integer. Works on arrays.
    Saves on parentheses, nothing more"""
    if iterable(n): # it's a sequence, return as an int64 array
        return np.int64(np.round(n))
    else: # it's a scalar, return as normal Python int
        return int(round(n))
            
def iterable(x):
    """Check if the input is iterable, stolen from numpy.iterable()"""
    try:
        iter(x)
        return True
    except:
        return False
            
