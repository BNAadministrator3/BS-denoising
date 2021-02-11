#This update increases the self.__decorate__() function to rectify the SNR. 
import math
import os 
import sys
sys.path.append('..')
from A2_collectingBS import getBSevents, visualizing
from Denoise_BS.B_process_BSevents import Normal_height_BS
from feature_transformation import read_wav_data, GetFrequencyFeatures 
import random
import numpy as np
import torch
import csv
import gc
import pdb
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import prettytable as pt
import copy
import pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# snrs = (0,-5,-10,-15,-20)
FAST = True
ROOT_SNRS = (0,-5,-10,-15,-20)
noise_set = ('Rub','Speech','Circuit noise','Snore','Cough','Groan','Instrument','Collision')
positive_dis = {'Rub':400,'Speech':400,'Circuit noise':400,'Snore':400,
                      'Cough':100,'Groan':100,'Instrument':100,'Collision':100}
negative_dis = {'Rub':400,'Speech':400,'Circuit noise':400,'Snore':400,
                      'Cough':100,'Groan':100,'Instrument':100,'Collision':100}

seed = 43
random.seed(seed)
np.random.seed(seed) 
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class get_noises():
    def __init__(self, noise_set,setpath=None, labelpath=None, snorepath=None):
        if setpath is None:
            setpath = os.path.join('../../','Data','BSfilteringData','data')
        self.setpath = setpath
        if labelpath is None:
            labelpath = os.path.join('../../','Data','BSfilteringData','weak_correction')
        self.labelpath = labelpath
        if snorepath is None:
            snorepath = os.path.join('../../','Data','BSfilteringData','1')
        self.snorepath = snorepath 
        self.load_all_noise(noise_set)
    
    def __load_one_noise_paths__(self, noise_type):
        noise_paths = []
        flag = noise_type.lower()
        label_file_paths = [os.path.join(self.labelpath, i) for i in os.listdir(self.labelpath) if 'xlsx' not in i]
        for label_file_path in label_file_paths:
            #1. detach the folder name
            foldername = os.path.split(label_file_path)[1]
            biname = 'bowels' if foldername[-5] == 'b' else 'non'
            foldername = foldername[0:-6]
            # pdb.set_trace()
            #2. get the seg path
            csv_reader = csv.reader(open(label_file_path))
            for row in csv_reader:
                decision = [ True for ele in row if flag in ele.lower() ]
                if len(decision) != 0:
                    assert(len(decision)==1)
                    noise_paths.append( (foldername,biname,row[0]) )
        return noise_paths
                # print(row)
                # pdb.set_trace()
                
    def __number_check__(self,noise_paths):
            keys = []
            for x,y,z in noise_paths:
                if x in keys:
                    pass
                else:
                    keys.append(x)
            ct = {}
            for key in keys:
                ct[key] = [0,0]
            for x,y,z in noise_paths:
                if y == 'bowels':
                    ct[x][0] = ct[x][0] + 1
                elif y == 'non':
                    ct[x][1] = ct[x][1] + 1
                else:
                    assert(False)
            tb = pt.PrettyTable() 
            tb.field_names = ["folder", "bowels", "non"]
            for key, values in ct.items():
                v1, v2 = values
                tb.add_row([key,v1,v2])
            print(tb)
            pdb.set_trace()
            
    def load_one_noise(self, noise_type):
        noise_segs = []
        paths = self.__load_one_noise_paths__(noise_type)
        # self.number_check(paths)
        for apath in paths:
            seg_path = os.path.join(self.setpath,apath[0],apath[1],apath[2])
            wavsignal, fs = read_wav_data(seg_path)
            wavsignal = wavsignal[0]
            output = (wavsignal,seg_path)
            noise_segs.append( output )
        return noise_segs

    def load_snore(self):
        snores = []
        wavpaths = [os.path.join(self.snorepath,i) for i in os.listdir(self.snorepath) if '.wav' in i]
        for wavpath in wavpaths:
            wavsignal, fs = read_wav_data(wavpath)
            wavsignal = wavsignal[0]
            snores.append( (wavsignal, wavpath) )
        return snores
 
    def load_all_noise(self, noise_set):
        noise_set = [ele.lower() for ele in noise_set]
        self.noises = {}
        for noise_type in noise_set:
            temp = self.load_one_noise(noise_type)
            self.noises[noise_type] = temp
        del self.noises['snore']
        self.noises['snore'] = self.load_snore()
        print('All noises are loaded.')
        # pdb.set_trace()
 
class get_bss():
    def __init__(self):
        self.seg_len = 4000
        bsnum = 0
        for key,value in positive_dis.items():
            bsnum = bsnum + value
        self.num = bsnum
        self.scope = [1,2,3,4,5]
    def bs_para(self, total, partition = None, scope = None, lbd = 1.3):
        if partition is None:
            partition = self.num
        if scope is None:
            scope = self.scope
        # pdb.set_trace()
        assert(min(scope)>=1)
        assert(isinstance(scope,list))
        assert(len(scope)>=3)
        assert(len(set(scope))==len(scope))
        a = np.array([scope,[1]*len(scope)]) 
        b = [total,partition]
        # pdb.set_trace()
        x = []
        for k in scope:
            if k != scope[-1]:
                pk = math.pow(lbd,k)/math.factorial(k)/2.718
                x.append(int(partition*pk))
            else:
                temp = partition - sum(x)
                x.append( temp )
        assert(len(x)==len(scope))
        # zx = [int(i) for i in x]
        # rest = partition - sum(zx)
        prac_use = sum([i * j for i,j in zip(x, scope)])
        # print('prac_use:',prac_use)
        # pdb.set_trace()
        assert(prac_use<=total)
        print('By caculation, a total of {} BS events shuold be used for dataset formation'.format(prac_use))
        final = []
        for scope_el, x_el in zip(scope,x):
            final = final + [scope_el] * x_el
        # final = final + [1] * rest
        print('*'*20)
        print('scope:',scope)
        print('solve',x)
        print('*'*20)
        check = [i for i in x if i<0]
        assert( len(check)==0 )
        return final
    
    def Len_sum(self,bs_seg):
        a = 0
        for bs in bs_seg:
            assert(isinstance(bs,list))
            a = a + len(bs)
        return a

    def bs_segmenting(self,bs_para,bs_events):
        bs_subdata = []
        start_point = 0
        for ele in bs_para:
            temp = bs_events[start_point:start_point+ele]
            # pdb.set_trace()
            bss=[]
            for ge,_ in temp:
                bss.append(ge)
            start_point = start_point + ele
            #2.1.1 blank bowel sound segments
            len_dif = self.seg_len - self.Len_sum(bss)
            lst = range(len_dif)
            bpoint = random.sample(lst,len(bss))
            bpoint.sort()
            bpoint = [0] + bpoint + [len_dif]
            py = []
            for i in range(1,len(bpoint)):
                temp = bpoint[i] - bpoint[i-1]
                py.append(temp)
            blank_bs = []
            bs_posi = []
            for id, bs in enumerate(bss):
                zeros = [0] * py[id]
                begin = len(blank_bs + zeros)
                blank_bs = blank_bs + zeros + bs
                ending = len(blank_bs)-1
                bs_posi.append((begin,ending))
            zeros = [0] * py[-1]
            blank_bs = blank_bs + zeros
            bs_subdata.append( (blank_bs,bs_posi) )
        return bs_subdata
    
class data_generator():
    def __init__(self,preprocess=None):
        if not FAST:
            temp = get_noises(noise_set)
            self.noises = temp.noises
            clean_events = self.Clean_bs_obtain()
            tmp = get_bss()
            paras = tmp.bs_para(len(clean_events))
            self.bowel_sounds = tmp.bs_segmenting(paras,clean_events)
            # get the standard amplitudes
            all_snrs = (20,15,10,5,0,-5,-10,-15,-20)
            all_snrs = set(all_snrs + ROOT_SNRS)
            self.standard_power(clean_events,all_snrs)
            self.positive_data = self.positive_gen_all(self.bowel_sounds,ROOT_SNRS)
            self.negative_data = self.negative_gen_all(self.bowel_sounds,ROOT_SNRS)
            self.data_dividing() #Finally output self.trains and self.tests
            self.__dump__( (self.trains, self.tests ))
            pdb.set_trace()
        else:
            self.trains, self.tests = self.__load__()
        self.data = self.trains + self.tests
        # self.trial = self.__positive_generate_one__(self.bowel_sounds[0:1000],('rub',1000),snrs)
        # self.trial = self.__negative_generate_one__(('rub',1000),snrs)
 
    def __load__(self,npath=None):
        if npath is None:
            npath = os.path.join( os.getcwd(), 'materials','Input_data0_-20.pkl' )
        with open(npath, 'rb') as n:
            data = pickle.load(n)
            print('A dataset is loaded.')
        return data
    
    def __dump__(self,data):
        target = os.path.join( os.getcwd(), 'materials','Input_data0_-20.pkl' )
        with open(target, 'wb') as f:
            pickle.dump(data, f)
        print('Data dumpped.')

    def standard_power(self,bs_events,all_snrs):
        joint = []
        for bs_event, _ in bs_events:
            joint = joint + list(bs_event)
        s = [np.float64(i) * np.float64(i) for i in joint]
        xpower = np.sum(s) / 1.0 / len(s)
        # pdb.set_trace()
        del s, joint
        gc.collect()
        noise_set = {}
        # seqs_len = self.seg_len * len(bs_events)
        self.standard_amplif = {}
        assert(0 in all_snrs)
        for snr in all_snrs:
            ratio = 10 ** (snr / 10.0)
            cor_power = xpower / ratio
            # a_noise = np.random.randn(seqs_len) * np.sqrt(noise_power)
            self.standard_amplif[snr] = np.sqrt( cor_power )
            
    def Clean_bs_obtain(self):
        bs_events = getBSevents()
        new_events = Normal_height_BS(bs_events)
        # seq,truth = visualizing( new_events, values=True )
        return new_events
 
    def __Smash__(self, big_seg):
        seg, info = big_seg
        new_segs = []
        for i in range(5):
            temp = seg[i*4000:(i+1)*4000]
            new_segs.append( (temp, info) )
        return new_segs
        
    def __beautify__(self, seg, snr_value):
        ampli, info = seg
        ss = np.float64(0)
        energy=sum([ np.float64(i)*np.float64(i) for i in ampli])
        try:
            assert(len(ampli) == 4000)
        except:
            pdb.set_trace()
        ss = energy/(len(ampli))
        sqrt_ss = np.sqrt(ss)
        k = self.standard_amplif[snr_value] / sqrt_ss
        new_ampli = [ i*k for i in ampli ]
        return (new_ampli,info)
    
    def __Sumchecking__(self, spvalue, dictionary):
        # pdb.set_trace()
        x = 0
        for key, value in dictionary.items():
            x =x + len(value)
        if x == spvalue:
            return True
        else:
            return False
    
    def __positioner__(self,position=None):
        if position is None:
            a = [(0,0),(0,0),(0,0),(0,0),(0,0)]
        else:
            diflen = 5 - len(position)
            sup = []
            for i in range(diflen):
                sup.append((0,0))
            position = list( position )
            a = position + sup
        a = np.array( a, dtype='int16' )
        return a
    
    def __decorate__(self, curve, spike, snr):
        Es = sum([np.float64(i)*np.float64(i) for i in spike])
        tpvalue = np.power(10, (snr/10.0))
        En = Es / tpvalue
        en = sum([np.float64(i)*np.float64(i) for i in curve])
        k = np.sqrt(En/en)
        new_curve = [np.float64(i)*np.float64(k) for i in curve]
        return new_curve
    
    def __positive_generate_one__(self,bowels,exp,snrs):
        assert(isinstance(exp,tuple))
        #specified number of segments, specified noise type, even for each snr
        exp = (exp[0].lower(),exp[1])
        quotient = exp[1] // len(snrs)
        remainder = exp[1] % len(snrs)
        base = [quotient] * len(snrs)
        sup = [1] * remainder + [0] * ( len(snrs)-remainder ) 
        amount = [i+j for i,j in zip(base,sup)]
        # 1.2 get the noise segments
        geshu = exp[1] // 5
        assert(exp[1] % 5 == 0)
        a_sort = self.noises[exp[0]]
        pick = random.choice(range(len(a_sort)))
        subset = []
        for i in range(geshu):
            point = i + pick
            if point>=len(a_sort):
                point = point % len(a_sort)
            tiles = self.__Smash__(a_sort[point])
            for tile in tiles:
                subset.append( tile )
        random.shuffle(subset)
        #1.3 normalize the noise segs to the specified snrs
        norm_noise = {}
        dict_bowel = {}
        noised_bowel = {}
        for snr in snrs:
            norm_noise[snr] = []
            dict_bowel[snr] = {}
            noised_bowel[snr] = []
        for i in range(len(amount)):
            stp = sum(amount[0:i])
            edp = sum(amount[0:i+1])
            segs = subset[stp:edp] #The bowel sound events used under each snr
            # pdb.set_trace()
            dict_bowel[snrs[i]] = bowels[stp:edp]
            for seg in segs:
                new_seg = self.__beautify__(seg, snrs[i])
                norm_noise[snrs[i]].append(new_seg)
        assert(self.__Sumchecking__(len(bowels),norm_noise))
        #1.4.2 superpose the norm_noise with bowel sound events
        for snr in snrs:
            for bn, nn in zip(dict_bowel[snr],norm_noise[snr]):
                # pdb.set_trace()
                spike, position = bn
                position = self.__positioner__(position)
                curve, info = nn
                curve = self.__decorate__(curve, spike, snr)
                synthesis = [i+j for i,j in zip(spike,curve)]
                if (spike == synthesis):
                    pdb.set_trace()
                noised_bowel[snr].append( ( synthesis,spike,(position,curve,(exp[0],snr),info) ) )
        print('Finish the building of BS segments within the {} noises.'.format(exp[0]))
        return noised_bowel
    
    def positive_gen_all(self,bowels,snrs):
        #1. assign bowels evenly
        random.shuffle(bowels)
        output = {}
        newbowels = copy.deepcopy(bowels)
        for key,value in positive_dis.items():
            temp = copy.deepcopy(newbowels[0:value])
            output[key] = self.__positive_generate_one__(temp,(key,value),snrs)
            del newbowels[0:value]
        return output
        
    def __negative_generate_one__(self,bowels,exp,snrs):
        assert(isinstance(exp,tuple))
        #specified number of segments, specified noise type, even for each snr
        exp = (exp[0].lower(),exp[1])
        aline = [0] * 4000
        quotient = exp[1] // len(snrs)
        remainder = exp[1] % len(snrs)
        base = [quotient] * len(snrs)
        sup = [1] * remainder + [0] * ( len(snrs)-remainder ) 
        amount = [i+j for i,j in zip(base,sup)]
        # 1.2 get the noise segments
        geshu = exp[1] // 5
        assert(exp[1] % 5 == 0)
        a_sort = self.noises[exp[0]]
        pick = random.choice(range(len(a_sort)))
        subset = []
        for i in range(geshu):
            point = i + pick
            if point>=len(a_sort):
                point = point % len(a_sort)
            tiles = self.__Smash__(a_sort[point])
            for tile in tiles:
                subset.append( tile )
        random.shuffle(subset)
        #1.3 normalize the noise segs to the specified snrs
        norm_noise = {}
        dict_bowel = {}
        noised_bowel = {}
        for snr in snrs:
            norm_noise[snr] = []
            dict_bowel[snr] = {}
            noised_bowel[snr] = []
        for i in range(len(amount)):
            stp = sum(amount[0:i])
            edp = sum(amount[0:i+1])
            segs = subset[stp:edp] #The bowel sound events used under each snr
            # pdb.set_trace()
            dict_bowel[snrs[i]] = bowels[stp:edp]
            for seg in segs:
                new_seg = self.__beautify__(seg, snrs[i])
                norm_noise[snrs[i]].append(new_seg)
        assert(self.__Sumchecking__(len(bowels),norm_noise))
        #1.4.2 superpose the norm_noise with bowel sound events
        for snr in snrs:
            for bn, nn in zip(dict_bowel[snr],norm_noise[snr]):
                # pdb.set_trace()
                spike, _ = bn
                position = self.__positioner__(None)
                curve, info = nn
                curve = self.__decorate__(curve, spike, snr)
                if (spike == curve):
                    pdb.set_trace()
                noised_bowel[snr].append( ( curve,aline,(position,curve,(exp[0],snr),info) ) )
        print('Finish the building of NBS segments within the {} noises.'.format(exp[0]))
        return noised_bowel
        
    def negative_gen_all(self, bowels, snrs):
        #1. assign bowels evenly
        output = {}
        # pdb.set_trace()
        # random.shuffle(bowels)
        for key,value in negative_dis.items():
            temp = copy.deepcopy(bowels[0:value])
            output[key] = self.__negative_generate_one__(temp,(key,value),snrs)
            del bowels[0:value]
        return output
        
    def data_dividing(self,ratio=0.8):
        self.trains = []
        self.tests = []
        for key,value in self.positive_data.items():
            for subkey,subvalue in self.positive_data[key].items():
                temp = self.positive_data[key][subkey]
                edpoint = int(len(temp) * ratio)
                if edpoint < len(temp) * ratio:
                    edpoint = edpoint + 1
                fortrain = temp[0:edpoint]
                fortest = temp[edpoint:]
                for atrain in fortrain:
                    self.trains.append( atrain )
                for atest in fortest:
                    self.tests.append( atest )
        for key,value in self.negative_data.items():
            for subkey,subvalue in self.negative_data[key].items():
                temp = self.negative_data[key][subkey]
                edpoint = int(len(temp) * ratio)
                if edpoint < len(temp) * ratio:
                    edpoint = edpoint + 1
                fortrain = temp[0:edpoint]
                fortest = temp[edpoint:]
                for atrain in fortrain:
                    self.trains.append( atrain )
                for atest in fortest:
                    self.tests.append( atest )
        random.shuffle( self.trains )
        random.shuffle( self.tests )
        # pdb.set_trace()
        print('Finish data dividing. A dataset is formed.')
        # for datum in self.trains:
            # if np.isnan(datum[1]).any():
                # pdb.set_trace()
        # for datum in self.tests:
            # if np.isnan(datum[1]).any():
                # pdb.set_trace()
    
class BSdataset(Dataset):
    def __init__(self, mydata, transform=None, preprocess = True):
        self.data = mydata
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data[idx][0]
        y = self.data[idx][1]
        x = np.array(x, dtype='float32')
        # z = self.data[idx][2]
        bs_flag = self.data[idx][2][0]
        noise_ori = self.data[idx][2][1]
        snr = self.data[idx][2][2][1] 
        if self.preprocess:
            x = x.reshape(1,x.shape[0],x.shape[1])
        else:
            pass
            # x = x.reshape(x.shape[0],1)
        if self.transform is not None:
            x = self.transform(x)
            x = np.array(x)
        y = np.array(y, dtype='float32')
        # z = np.array(z, dtype='float32')
        # w = np.array(int(w), dtype=''
        # y = y.reshape(y.shape[0], 1)
        # pdb.set_trace()
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(bs_flag), torch.tensor(noise_ori), torch.tensor(snr)
    
    def break_down(self):
        x = []
        y = []
        z = []
        w = []
        for a, b, c in self.data:
            a = np.array(a, dtype='float32')
            if self.preprocess:
                a = a.reshape(1,a.shape[0],a.shape[1])
            else:
                a = a.reshape(a.shape[0],1)
            b = np.array(b, dtype='int16')
            # b = b.reshape(b.shape[0], 1)
            x.append(a)
            y.append(b)
            z.append(c)
            # w.append(d)
        return torch.from_numpy(np.array(x, dtype='float32')), torch.from_numpy(np.array(y, dtype='int16')), z

    def Normal_getitem(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data[idx][0]
        y = self.data[idx][1]
        z = self.data[idx][2]
        x = np.array(x, dtype='float32')
        if self.preprocess:
            x = x.reshape(1,x.shape[0],x.shape[1])
        else:
            pass
        # y = np.array(y, dtype='int16')
        # y = y.reshape(y.shape[0], 1)
        return x, y, z
        
# if __name__ == '__main__':
    # a = data_generator()
    # okay = a.data
    # b = BSdataset(okay,preprocess = False)
    # train_loader = torch.utils.data.DataLoader(
        # b,
        # batch_size=32, shuffle=True,
        # num_workers=8, pin_memory=True)
    # for id,(x,y,z) in enumerate(train_loader):
        # pdb.set_trace()
    # for idx in range(b.__len__()):
        # cc,vv,bb = b.__getitem__(idx)
        # assert( len(bb) == 4 )
        # assert(vv.shape[0] == 4000)

if __name__ == '__main__':
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
            add = ori
            plt.subplot(411)
            plt.plot(he)
            plt.subplot(412)
            plt.plot(ori)
            plt.subplot(413)
            plt.plot(add)
            stft = GetFrequencyFeatures(he, 4000) 
            plt.subplot(414)
            plt.imshow(stft)
            plt.suptitle(tstr, fontsize=14)
            plt.show()
            i = i + 1
            if i == len(ROOT_SNRS):
                i = 0
            
            pdb.set_trace()
        
    for j in range(len(okay[ROOT_SNRS[0]])):
        he = okay[ROOT_SNRS[i]][j][0]
        ori = okay[snrs[i]][j][1]
        add = okay[snrs[i]][j][2][1]
        add = ori
        plt.subplot(411)
        plt.plot(he)
        plt.subplot(412)
        plt.plot(ori)
        plt.subplot(413)
        plt.plot(add)
        stft = GetFrequencyFeatures(he, 4000) 
        plt.subplot(414)
        plt.imshow(stft)
        plt.suptitle(str(snrs[i]), fontsize=14)
        plt.show()
        i = i + 1
        if i == len(snrs):
            i = 0
        
        pdb.set_trace()
 # a = get_noises()
    # rubs = a.load_one_noise(noise_set[0])
    # for i in rubs:
        # plt.plot(i[0])
        # plt.show()
        # pdb.set_trace()
     