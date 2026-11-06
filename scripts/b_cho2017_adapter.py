"""Cho2017 structural adapter; input .mat, output own-CAR3 3-second epochs."""
import argparse
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter,sosfiltfilt,resample_poly

def adapt(path):
    eeg=loadmat(path,squeeze_me=True,struct_as_record=False)['eeg']
    fs=int(eeg.srate)
    assert fs==512
    marker=np.asarray(eeg.imagery_event).ravel()
    onsets=np.flatnonzero((marker!=0)&np.r_[True,marker[:-1]==0])
    assert len(onsets)>=45
    epochs=[];labels=[];indices=[]
    for label,name in enumerate(('imagery_left','imagery_right')):
        raw=np.asarray(getattr(eeg,name),dtype=float)
        assert raw.ndim==2 and raw.shape[0]>=64 and raw.shape[1]==len(marker)
        # Documented acquisition order: C3, Cz, C4.
        x=raw[[12,47,49]]
        x=x-x.mean(0,keepdims=True)
        x=sosfiltfilt(butter(4,[.3,40],fs=fs,btype='bandpass',output='sos'),x,axis=-1)
        x=resample_poly(x,25,64,axis=-1)/100
        for event,start in enumerate(onsets):
            begin=int(round(start*200/fs))
            trial=x[:,begin:begin+600]
            assert trial.shape==(3,600) and np.isfinite(trial).all()
            epochs.append(trial.reshape(3,3,200));labels.append(label);indices.append(event)
    return dict(x=np.asarray(epochs,dtype=np.float32),y=np.array(labels),event=np.array(indices))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    assert not args.output.exists()
    np.savez_compressed(args.output,**adapt(args.input))
