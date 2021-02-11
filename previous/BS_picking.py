from snr_computing import read_in_single_textgrid, extract_interval
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from feature_transformation import * 
import pdb
# import tgt
if __name__=="__main__":
    # src = os.path.join('I:\\thudata','all_filtered_data_&&_alittle_routines','filtered')
    src = os.path.join('I:\\thudata','doctorsdata','001')
    # src1 = os.path.join(src,'fed','20190525')
    # trial = os.path.join(src,src1,'006')
    point = '175473075224640_2019_05_25_09_50_48_m1.TextGrid'
    print(point)
    # tgfile = os.path.join(trial, point)
    tgfile = os.path.join(src, point)
    #output 1
    tgdata = read_in_single_textgrid(tgfile)
    info = extract_interval(tgdata.tiers[0].intervals)
    # print(info)
    audiofile = tgfile.replace('_m1.TextGrid','.wav')
    #output 2
    wavsignal, fs = read_wav_data(audiofile)
    wavsignal = wavsignal[0]
    bs_events = []
    for mark in info:
        if mark[2] == 'T' or mark[2] == 'ST':
            print(mark)
            st = myround(mark[0]*fs)
            ed = myround(mark[1]*fs)
            bs_events.append(wavsignal[st:ed])
    print(len(bs_events))
    line = [0 for _ in range(200)]
    altogether = []
    for i, bs_event in enumerate(bs_events):
        altogether = altogether + line
        # pdb.set_trace()
        altogether = altogether + bs_event.tolist()
        if (i+1) == len(bs_events):
            altogether = altogether + line
    plt.plot(altogether)
    plt.show()
    