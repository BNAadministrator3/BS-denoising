# Replicate the IMF-FD work
import os 
import math
import pdb
import sys
from PyEMD import EMD,Visualisation
import numpy as np
from A_1_0_op_geData import data_generator, BSdataset, ROOT_SNRS
import matplotlib.pyplot as plt
import gc

############
import torch
from hht import hilbert_huang, hilbert_spectrum, plot_IMFs, _shrink
os.environ['CUDA_VISIBLE_DEVICES']=''
############

def Frac_dimension(seq):
    wl = 25
    assert(wl%2==1)
    scope = len(seq) -wl + 1 
    fd_seq = []
    for i in range(scope):
        piece = seq[0+i:wl+i]
        lc = 0
        for j in range(wl-1):
            ver = abs( piece[j+1] - piece[j] )
            lc = lc + np.sqrt( np.float64(ver) * np.float64(ver) + 0.0625*1e-6  )
        try:
            fd = 1 +  math.log(lc) / (  math.log(2) +  math.log(wl-1) )
        except:
            # plt.plot(seq)
            # plt.show()
            pdb.set_trace()
        fd_seq.append( fd )
    geshu = (wl - 1) // 2 
    fd_seq = [0] * geshu + fd_seq + [0] * geshu
    assert(len(fd_seq) == len(seq))
    return np.array(fd_seq)

def Imf_fd(seq):
    s = np.array(seq)
    emd = EMD()
    IMFs = emd(s)
    assert(len(IMFs)>=7)
    #2. sifting
    pick = IMFs[1:6]
    #3. get fds
    fds = []
    for imf in pick:
        temp = Frac_dimension(imf)
        fds.append( temp )
    #4. peeling
    pImfs = []
    for afd, aimf in zip(fds,pick):
        # mu = np.mean(afd)
        pim = []
        sigma = np.std(afd,ddof=1)
        for id, value in enumerate(afd):
            if value + 0.3216 > sigma:
                pim.append( aimf[id] )
            else:
                pim.append( 0 )
        pImfs.append( pim )
    pImfs = np.array(pImfs)
    result = np.zeros( len(seq) )
    for i in range(len(seq)):
        result[i] = sum( pImfs[:,i] )
        # pdb.set_trace()
    return result
    
def Complete_imf_fd(seq):
    s = np.array(seq)
    emd = EMD()
    IMFs = emd(s)
    assert(len(IMFs)>=7)
    #2. get fds
    fds = []
    for imf in IMFs:
        temp = Frac_dimension(imf)
        fds.append( temp )
    #3. sifting
    muset = []
    sigmaset = []
    for afd in fds:
        muset.append( np.mean(afd) )
        sigmaset.append( np.std(afd,ddof=1) )
    # pdb.set_trace()
    mu_thr = np.mean( muset ) + np.std( muset,ddof=1 )
    sigma_thr = np.mean( sigmaset ) + np.std( sigmaset,ddof=1 )
    pick = []
    for i in range( len(IMFs) ):
        if muset[i] >= mu_thr and sigmaset[i] >= sigma_thr:
            pick.append(IMFs[i])
    if len(pick) == 0:
        print('None of the imfs are picked, so we use the already-got results.')
        pick = IMFs[1:6]
    else:
        print('{} levels are got.'.format(len(pick)))
    #4. peeling
    pImfs = []
    for afd, aimf in zip(fds,pick):
        # mu = np.mean(afd)
        pim = []
        sigma = np.std(afd,ddof=1)
        for id, value in enumerate(afd):
            # if value + 0.3216 > sigma:
            if value - min(afd) > sigma:
                pim.append( aimf[id] )
            else:
                pim.append( 0 )
        pImfs.append( pim )
    pImfs = np.array(pImfs)
    result = np.zeros( len(seq) )
    for i in range(len(seq)):
        result[i] = sum( pImfs[:,i] )
        # pdb.set_trace()
    return result
    
def Visualing(peeled_seq, fs=4000):
    t = np.arange(0,1,1/fs)
    # print(len(t))
    emd = EMD()
    emd.emd(peeled_seq)
    imfs, res = emd.get_imfs_and_residue()
    vis = Visualisation(emd)
    # Create a plot with all IMFs and residue
    vis.plot_imfs(imfs=imfs, residue=res, t=t, include_residue=True)
    # Create a plot with instantaneous frequency of all IMFs
    vis.plot_instant_freq(t, imfs=imfs)
    # Show both plots
    vis.show()
    
def Inst_freq(peeled_seq, fs=4000):
    t = np.arange(0,1,1/fs)
    # print(len(t))
    emd = EMD()
    emd.emd(peeled_seq)
    imfs, res = emd.get_imfs_and_residue()
    vis = Visualisation(emd)
    imfs_inst_freqs = vis._calc_inst_freq(imfs, t, order=False, alpha=None)
    return imfs_inst_freqs
    
def HHT_spectrum(peeled_seq, fs=4000):
    delta_t = 1 / fs
    peeled_seq = torch.tensor(peeled_seq)
    imfs, imfs_env, imfs_freq = hilbert_huang(peeled_seq, delta_t, num_extrema=3)
    spectrum, t, f = hilbert_spectrum(imfs_env, imfs_freq, delta_t, time_range = (0.1, 0.9))
    return spectrum, t, f

def AE(peeled_seq, fs=4000):
    delta_t = 1 / fs
    peeled_seq = torch.tensor(peeled_seq)
    imfs, imfs_env, imfs_freq = hilbert_huang(peeled_seq, delta_t, num_extrema=3)
    spectrum, time_axis, freq_axis = hilbert_spectrum(imfs_env, imfs_freq, delta_t, time_range = (0.1, 0.9), display=False)
    spectrum_, time_axis_, freq_axis_ = _shrink(spectrum, time_axis, freq_axis) #Decimation!
    spectrum_ = spectrum_.cpu().to_dense().numpy()
    rotation = spectrum_.T
    time_axis_ = time_axis_.cpu()
    freq_axis_ = freq_axis_.cpu()
    del spectrum, time_axis, freq_axis
    gc.collect()
    p0_80 = 0 #left closed and right open
    p80_1000 = 0
    p100_300 = 0
    p300_1000 = 0
    # assert(freq_axis_[-1]>300)
    for time_id in range( rotation.shape[1] ):
        for freq_id in range( rotation.shape[0] ):
            value = rotation[freq_id,time_id]
            time_point = time_axis_[time_id]
            freq_point = freq_axis_[freq_id]
            if freq_point>=0 and freq_point<80:
                p0_80 = p0_80 + value
            if freq_point>=80 and freq_point<1000:
                p80_1000 = p80_1000 + value
            if freq_point>=100 and freq_point<300:
                p100_300 = p100_300 + value
            if freq_point>=300 and freq_point<1000:
                p300_1000 = p300_1000 + value
    if p80_1000!=0:
        R0_80_1000 = p0_80 / p80_1000 
    else: 
        R0_80_1000 = 1e5
    if p300_1000!=0:
        R100_300_1000 = p100_300 / p300_1000
    else:
        R100_300_1000 = 1e5
    return R0_80_1000, R100_300_1000 
    # eps = spectrum_.max() * (1e-5)
    # coutour = plt.pcolormesh(time_axis_.cpu(), freq_axis_.cpu(), 10*np.log10(spectrum_.T + eps), cmap = plt.cm.YlGnBu_r)
    

if __name__ == '__main__':
    # s = np.random.random(100)
    a = data_generator()
    okay = a.data
    i = 0
    for datum in a.data:
        if datum[2][2][1] == ROOT_SNRS[i]:
            he = datum[0]
            ori = datum[1]
            if np.sum(datum[2][0]) == 0: 
                label = 'NBS'
            else:
                label = 'BS'
            tstr = label + '_' + datum[2][2][0] + '_' + str(datum[2][2][1])
            s = np.array(he)
            # x = Imf_fd(s)
            y = Complete_imf_fd(s)
            plt.subplot(311)
            plt.plot(he)
            plt.subplot(312)
            plt.plot(ori)
            plt.subplot(313)
            plt.plot(y)
            plt.suptitle(tstr, fontsize=14)
            plt.show()
            i = i + 1
            if i == len(ROOT_SNRS):
                i = 0
            # emd = EMD()
            # IMFs = emd(s)
            # ddd = Visualing(y)
            spectrum, t, f = HHT_spectrum(y)
            r1, r2 = AE(y)
            pdb.set_trace()