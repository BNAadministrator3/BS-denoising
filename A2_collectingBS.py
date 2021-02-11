import os
import pdb
from previous.snr_computing import read_in_single_textgrid, extract_interval
from feature_transformation import *
import matplotlib.pyplot as plt

def visualizing(bs_events,values=False,loc=False):
    # line = [0 for _ in range(200)]
    line = [0 for _ in range(100)]
    altogether = []
    truth = []
    for i, bs_event in enumerate(bs_events):
        altogether = altogether + line
        # pdb.set_trace()
        bs_event = list(bs_event)
        if not isinstance(bs_event[0],list):
            bs_event[0] = bs_event[0].tolist()
        altogether = altogether + bs_event[0]
        ed = len(altogether) - 1
        st = ed - len(bs_event[0]) + 1
        truth.append((st,ed))        
        if (i+1) == len(bs_events):
            altogether = altogether + line
    if values:
        print('A BS sequence is concatenated.')
        return altogether, truth
    plt.plot(altogether)
    if loc:
        print()
        for i, bs_event in enumerate(bs_events):
            print('{}, its location is: {}'.format(i+1,bs_event[1]))
    plt.show()

def getBSevents():
    # /home/zhaok14/example/Data/DenoiseBSdata/
    rdir = os.path.join( '/home','zhaok17','MyProjects','Data','DenoiseBSdata') 
#    rdir = os.path.join( '/home','zhaok14','example','Data','DenoiseBSdata') 
#    my = os.path.join(rdir, 'all_filtered_data_&&_alittle_routines','filtered')
    my = os.path.join(rdir,'filtered')
    doctor = os.path.join(rdir, 'doctorsdata')
    brother = os.path.join(rdir, 'yuantaodata')

    #Step 1. extract my BSs
    fast0525 = os.path.join(my,'fast','20190525')
    fast0526 = os.path.join(my,'fast','20190526')
    fed = os.path.join(my,'fed','20190525')
    dirset = [(fast0525,'fast_0525'),(fast0526,'fast_0526'),(fed,'fed_0525')]
    bs_events = []
    ct_events = []
    for adir, dirc in dirset:
        folders = [ (os.path.join(adir,i),i) for i in os.listdir(adir) if os.path.isdir( os.path.join(adir,i) )]
        for folder_path, folder in folders:
            # for i in os.listdir(folder_path):
            #     if '_m1' in i:
            #         print(i)
            #         pdb.set_trace()
            files = [ (os.path.join(folder_path,i), i[-20:-12])  for i in os.listdir(folder_path) if '_m1' in i]
            for file_path, date in files:
                tgdata = read_in_single_textgrid(file_path)
                info = extract_interval(tgdata.tiers[0].intervals)
                audiofile = file_path.replace('_m1.TextGrid', '.wav')
                # output 2
                wavsignal, fs = read_wav_data(audiofile)
                wavsignal = wavsignal[0]
                for mark in info:
                    if mark[2] == 'CT':
                        st = myround(mark[0] * fs)
                        ed = myround(mark[1] * fs)
                        ct_events.append((wavsignal[st:ed],0))
                    if mark[2] == 'T' or mark[2] == 'ST':
                        # print(mark)
                        if mark[1]-mark[0]<0.1:  #Exclude the multiple bursts
                            # 1). bs events
                            st = myround(mark[0] * fs)
                            ed = myround(mark[1] * fs)
                            # 2). time stamps
                            tstamp = (round(mark[0],2), round(mark[1],2))
                            location = (dirc, folder, date, tstamp)
                            bs_events.append((wavsignal[st:ed],location))
    print('The BS events annotated on my own are up to: ',len(bs_events))                
    # visualizing(ct_events)
    # pdb.set_trace()

    #Step 2. extract the doctor's BSs
    doctor = os.path.join(rdir,'doctorsdata')
    del folders
    folders = [ (os.path.join(doctor,i),i) for i in os.listdir(doctor) if os.path.isdir( os.path.join(adir,i) )]
    dirc = 'fed_0525'
    for folder_path, folder in folders:
        # pdb.set_trace()
        files = [ (os.path.join(folder_path,i), i[-20:-12])  for i in os.listdir(folder_path) if '_m1' in i]
        for file_path, date in files:
            tgdata = read_in_single_textgrid(file_path)
            info = extract_interval(tgdata.tiers[0].intervals)
            audiofile = file_path.replace('_m1.TextGrid', '.wav')
            # output 2
            wavsignal, fs = read_wav_data(audiofile)
            wavsignal = wavsignal[0]
            for mark in info:
                if mark[2] == 'T':
                    # print(mark)
                    if mark[1]-mark[0]<0.1:  #Exclude the multiple bursts
                        # 1). bs events
                        st = myround(mark[0] * fs)
                        ed = myround(mark[1] * fs)
                        # 2). time stamps
                        tstamp = (round(mark[0],2), round(mark[1],2))
                        location = (dirc, folder, date, tstamp)
                        bs_events.append((wavsignal[st:ed],location))
        # print(len(bs_events))
        # pdb.set_trace()
    print("After adding the doctor's data, the BS events are up to:",len(bs_events)) 

    #Step 3. extract the expert's data
    dirc = 'fast_0526'
    expert_labels = os.path.join(rdir,'yuantaodata')
    expert_wavfiles = os.path.join(my, 'fast', '20190526')
    curbs = ('526006','526007','526008','526009','526011','526012')
    for curb in curbs:
        wavfiles = os.path.join(expert_wavfiles,curb)
        labels = os.path.join(expert_labels,curb)
        yesican = [(os.path.join(labels,i),i[-20:-12],i.replace('_m1.TextGrid', '.wav'))  for i in os.listdir(labels) if '_m1' in i]
        for tgfile_path, date, wavfile in yesican:
            tgdata = read_in_single_textgrid(tgfile_path)
            info = extract_interval(tgdata.tiers[0].intervals)
            audiofile = os.path.join(wavfiles,wavfile)
            # audiofile = file_path.replace('_m1.TextGrid', '.wav')
            # output 2
            wavsignal, fs = read_wav_data(audiofile)
            wavsignal = wavsignal[0]   
            for mark in info:
                if mark[2] == 'T':
                    # print(mark)
                    if mark[1]-mark[0]<0.1:  #Exclude the multiple bursts
                        # 1). bs events
                        st = myround(mark[0] * fs)
                        ed = myround(mark[1] * fs)
                        # 2). time stamps
                        tstamp = (round(mark[0],2), round(mark[1],2))
                        location = (dirc, curb, date, tstamp)
                        bs_events.append((wavsignal[st:ed],location))
    print("After adding the expert's data, the BS events are up to:",len(bs_events)) 
    # return bs_events

#sifting BS
    sifted_bs_events = []
    for event,loc in bs_events:
        # pdb.set_trace()
        small_width = abs( np.where(event == np.max(event))[0][0] - np.where(event == np.min(event))[0][0] )
        big_width = len(event)
        if small_width<=15 and big_width<150:
            sifted_bs_events.append( (event,loc) )
    
    print("After screening, the BS events are down to:",len(sifted_bs_events)) 
    return sifted_bs_events




