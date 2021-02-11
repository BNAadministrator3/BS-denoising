import os 
import torch.nn as nn
from A_1_0_op_geData import data_generator, BSdataset
from A_1_0_op_geData import ROOT_SNRS
from B_0_0_IMF_FD import Complete_imf_fd, AE
import torch
import time
import numpy as np
import gc
from torchsummary import summary
import sys
sys.path.append('..')
from Denoise_BS.utils.utils import *
from Denoise_BS.utils.visuals import presentor
from thop import profile
from prettytable import PrettyTable
from math import log10 as log
import matplotlib.pyplot as plt
import pdb
eps = 1e-5
os.environ['CUDA_VISIBLE_DEVICES']=''

class logger(object):
    def __init__(self,filename=None):
        if filename is None:
            filename = os.path.join(os.getcwd(), 'Default.log')
        self.terminal = sys.stdout
        self.log = open(filename,'w')
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


result = [466,-512,62]
def Thresholding(seq, peak = result[0], trough = result[1], width = result[2]):
    position = [] # The elements of the list is a tuple
    for st,value in enumerate(seq):
        if ( len(position) > 0 ) and ( st in range(position[-1][0], position[-1][1]+1) ):
            continue
        flag = False
        if value >= peak:
            clip = seq[st+1:st+width+1]
            for ofset, point in enumerate(clip):
                if point <= trough:
                    flag = True
                    break
        elif value <= trough:
            clip = seq[st+1:st+width+1]
            for ofset, point in enumerate(clip):
                if point >= peak:        
                    flag = True
                    break
        if flag:
            position.append( (st, st+1+ofset) )
    return position

def accuracy_pure(logits_student, target, truth, fn=Thresholding):
    #Related works, fn could be artifact elimination
    #1. detect the filtered sequence
    xs = logits_student
    pred = []
    for x,y in zip(xs, target):
        r1,r2 = AE(x)
        if r1>0.5 or r2<0.5:
            pred.append( 0 )
        else:
            pred.append( 1 )
    correct = sum([i==j for i,j in zip(pred,truth)])
    batch_size = target.shape[0]
    accu = correct / batch_size * 100.0
    return accu    

def accuracy(logits_student, target, truth, fn=Thresholding):
    #Related works, fn could be artifact elimination
    #1. detect the filtered sequence
    xs = logits_student
    pred = []
    for x,y in zip(xs, target):
        r1,r2 = AE(x)
        if r1>0.5 or r2<0.5:
            pred.append( 0 )
        else:
            position = fn(x)
        # com = fn(y)
        # pdb.set_trace()
        # assert(len(position) == len(com))
            if len(position)!=0:
                pred.append( 1 )
            else:
                pred.append( 0 )
    correct = sum([i==j for i,j in zip(pred,truth)])
    batch_size = target.shape[0]
    accu = correct / batch_size * 100.0
    return accu    

def __SNRimproving__(xs, target, noise_ori):
    #1. get the residual noises after filtering
    residue = [i-j for i,j in zip(xs, target)]
    power_residue = sum(np.float64(i) * np.float64(i) for i in residue)/len(residue)
    log_pow_resi = 10 * log( power_residue + eps)
    power_ori = sum(np.float64(i) * np.float64(i) for i in noise_ori) / len(noise_ori)
    log_pow_ori = 10 * log( power_ori + eps)
    diff = log_pow_ori - log_pow_resi
    return diff
    
def SNRDif(logits_student, target, noise_ori, snrs ,truth ):
    truth = [ i == 1 for i in truth ]
    logits_student = logits_student[truth]
    target = target[truth]
    noise_ori = noise_ori[truth]
    snrs = snrs[truth]
    num = logits_student.shape[0]
    res = {}
    keys = set(snrs.tolist())
    # pdb.set_trace()
    for key in keys:
        res[key] = []
    # disc = map( __SNRimproving__,logits_student, target, noise_ori)
    disc = []
    for i in range(num):
        # pdb.set_trace()
        temp = __SNRimproving__(logits_student[i], target[i], noise_ori[i])
        disc.append( temp )
    for key,value in zip(snrs,disc):
        res[key] = value
    return res

def Z_Score(data):
    if sum(data) != 0:
        data = np.array(data)
        ave = np.mean(data)
        std = np.std(data,ddof=1)
        new_data = [(i-ave)/std for i in data]
    else:
        new_data = data
    return new_data

def __Correlation__(logits_student, target):
    xs = Z_Score(logits_student)
    target = Z_Score(target)
    pearson =  sum(i*j for i,j in zip(xs,target))    
    pearson = pearson / len(xs)
    assert(abs(pearson)<=1)
    return pearson

def Corr(logits_student, target):
    num = logits_student.shape[0]
    crs = []
    for i in range(num):
        # temp = Correlation( target[i], target[i] )
        temp = __Correlation__( logits_student[i], target[i] )
        crs.append( temp )
    return np.mean(crs)
            
def measure_val(logits_student, target, refill, fn=Thresholding):
    with torch.no_grad():
        logits_student = logits_student.detach().cpu().numpy()
        target = target.cpu().numpy()
        bsflag, noise_ori, batch_snr = refill
        bsflag = bsflag.cpu().numpy()
        noise_ori = noise_ori.cpu().numpy()
        batch_snr = batch_snr.cpu().numpy()
        # pdb.set_trace()
        truth = []
        for z in bsflag:
            if np.sum(z) == 0:
                truth.append( 0 )
            else:
                truth.append( 1 )
        accur = accuracy(logits_student, target, truth, fn=fn)
        pure = accuracy_pure(logits_student, target, truth, fn=fn)
        corr = Corr(logits_student, target)
        impr = SNRDif(logits_student, target, noise_ori, batch_snr, truth)
        return accur, pure, corr, impr
 
def validate(epoch, val_loader, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top2 = AverageMeter('AccPure@2', ':6.2f')
    top5 = AverageMeter('Cor@5', ':6.2f')
    top9 = MyAverageMeter(ROOT_SNRS, ':6.2f') #Note snrs depend on the import module
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top2, top5],
        prefix='Test: ')

    # switch to evaluation mode
    with torch.no_grad():
        end = time.time()
        for i, (images, target, mm,nn,kk) in enumerate(val_loader):
            refill = (mm,nn,kk)
            #1. change images to a ndarray
            # pdb.set_trace()
            x_seq = np.squeeze(images.cpu().numpy())
            filtered = Complete_imf_fd( x_seq )
            x_pred = torch.tensor(filtered).view(1,4000).to(device)
            target = target.to(device)
            # measure accuracy and record loss
            loss = criterion(x_pred, target.float())
            prec1, pure, corr, impr = measure_val(x_pred, target, refill)
            n = images.size(0)
            losses.update(loss.item(), n)
            top1.update(prec1, n)
            top2.update(pure, n)
            top5.update(corr, n)
            top9.update(impr, n)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            progress.display(i)

        x = PrettyTable(field_names=["ROOT_SNRS", "Improv."])
        imean = top9.Avg()
        for key, value in imean.items():
            x.add_row([key,value])
        print(x)
        del x 
        gc.collect() 
        print(' * acc@1 {top1.avg:.3f} AccPure@2 {top2.avg:.3f} cor@5 {top5.avg:.3f}'
              .format(top1=top1, top2=top2, top5=top5))
              
    return losses.avg, top1.avg #, top5.avg  
    
         
if __name__ == '__main__':
    #1.1 Setup: GPU, hyperparameters, optimizers and loss
    sys.stdout = logger(filename=os.path.join(os.getcwd(),'materials','EMD+FC+AE.log'))
    start_t = time.time()
    use_cuda = False
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    print('This time the {} is used.'.format(device))
    EPOCH = 1
    workers=40
    criterion = nn.MSELoss()
    
    #1.2 Build the dataset
    a = data_generator(preprocess = False)
    tests = BSdataset(a.tests,preprocess = False)
    # pdb.set_trace()
    val_loader = torch.utils.data.DataLoader(
        tests,
        batch_size=1, shuffle=False,
        num_workers=workers, pin_memory=True)
    
    #2 Training begins
    st = time.time()
    for epoch in range(EPOCH):
        valid_obj, valid_top1_acc = validate(epoch, val_loader, criterion)
    training_time = (time.time() - start_t) / 3600
    print('total valiadation time = {} hours'.format(training_time))
    
    # pdb.set_trace()

