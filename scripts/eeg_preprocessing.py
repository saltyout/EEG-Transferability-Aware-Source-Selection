"""Cho2017 structural adapter; input .mat, output own-CAR3 3-second epochs."""
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly

def cho_epochs(path):
    eeg=loadmat(path,squeeze_me=True,struct_as_record=False)['eeg']
    fs=int(eeg.srate)
    if fs != 512: raise ValueError("Cho2017 requires 512 Hz")
    marker=np.asarray(eeg.imagery_event).ravel()
    onsets=np.flatnonzero((marker!=0)&np.r_[True,marker[:-1]==0])
    if len(onsets) not in (100,120): raise ValueError("Expected 100 or 120 shared task onsets")
    epochs=[];labels=[];indices=[]
    for label,name in enumerate(('imagery_left','imagery_right')):
        raw=np.asarray(getattr(eeg,name),dtype=float)
        if raw.ndim != 2 or raw.shape[0] != 68 or raw.shape[1] != len(marker) or not np.isfinite(raw).all():
            raise ValueError("Expected finite original 68-channel Cho2017 recordings")
        # Documented acquisition order: C3, Cz, C4.
        x=raw[[12,47,49]]
        x=x-x.mean(0,keepdims=True)
        x=sosfiltfilt(butter(4,[.3,40],fs=fs,btype='bandpass',output='sos'),x,axis=-1)
        x=resample_poly(x,25,64,axis=-1)/100
        for event,start in enumerate(onsets):
            begin=int(round(start*200/fs))
            trial=x[:,begin:begin+600]
            if trial.shape != (3,600) or not np.isfinite(trial).all():
                raise ValueError("Invalid Cho2017 task window")
            epochs.append(trial.reshape(3,3,200));labels.append(label);indices.append(event)
    return dict(x=np.asarray(epochs,dtype=np.float32),y=np.array(labels),event=np.array(indices))
