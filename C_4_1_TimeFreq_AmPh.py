#This file is about a u-net followed by a time-domain neural networks
import os 
import torch.nn as nn
from A_2_0_Freq_OpGeData import data_generator, BSdataset_recovery
from A_2_0_Freq_OpGeData import ROOT_SNRS
from torch.utils.data import Dataset
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
from feature_transformation import istft 
import pdb
eps = 1e-5

class BSdataset_trch(Dataset):
    def __init__(self, mydata, transform=None, preprocess = True):
        self.data = self.Spec_forming( mydata )
        self.transform = transform
        self.preprocess = preprocess

    def Spec_forming(self, data, fs=4000):
        new_data = []
        for x, y, z in data:
            y = torch.tensor( y , dtype = torch.float32 ) 
            # pdb.set_trace()
            y_complex = torch.stft(y, n_fft=128, return_complex=True )
            y_abs = torch.abs( y_complex )
            # print(x_abs.shape)
            # print(x_ang.shape)
            # pdb.set_trace()
            # y = GetFrequencyFeatures(y, fs)
            # p,q,m,r = z
            # q = GetFrequencyFeatures(q, fs)
            # assert( isinstance( m, tuple ) )
            # z = ( p,q,m,r )
            new_data.append( (x,(y_abs, y), z) )
            # pdb.set_trace()
        print('ALL the spectrograms are generated within their phase info.')
        return new_data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data[idx][0]
        y_abs, y = self.data[idx][1]
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
        return torch.from_numpy(x), y_abs, y, torch.from_numpy(bs_flag), torch.tensor(noise_ori), torch.tensor(snr)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out,Size=None,Scale_factor=None):
        super(up_conv,self).__init__()
        if Size is not None and Scale_factor is not None:
            print('Error!')
            assert('False')
        elif Size is not None:
            self.up = nn.Sequential(
            nn.Upsample(size=Size),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
            )
        elif Scale_factor is not None:
            self.up = nn.Sequential(
            nn.Upsample(scale_factor=Scale_factor),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
            )
        else:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True)
            )

    def forward(self,x):
        x = self.up(x)
        return x

class U_NetCom(nn.Module):
    def __init__(self,img_ch=1,output_ch=1):
        super(U_NetCom,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512, Size=(8,15))
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256, Size=(16,31))
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128, Size=(32,63))
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64, Size=(65,126))
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)
        
        self.Fine_tune = nn.Linear(4000,4000)

    def forward(self,x):
        # encoding path
        temp = torch.stft( x, n_fft=128, return_complex=True )
        x_abs = torch.abs(temp)
        x_ang = torch.angle(temp)
        x_abs = torch.unsqueeze(x_abs, dim=1)
        
        # pdb.set_trace() #Get the right size!
        # x_abs = x_abs.reshape( x_abs.shape[0],1, 65, 126 )
        
        x1 = self.Conv1(x_abs)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        # pdb.set_trace()
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        d1 = torch.squeeze(d1)
        
        S_complex = d1 * torch.exp(1j * x_ang)
        synthesis = torch.istft( S_complex, n_fft=128 ) # The istft is not the same as that of librosa!
        
        # out = self.Fine_tune( synthesis )
        out = synthesis
        
        return d1,out

class BiLoss(nn.Module):
    def __init__(self):
        super(BiLoss, self).__init__()
        self.critical = nn.MSELoss()
        
    def forward(self, prob0, prob1, truth0, truth1):
        los1 = self.critical(prob0, truth0)
        los2 = self.critical(prob1, truth1) * 55 #You also need to change the factor!
        loss = ( los1 + los2 ) / 2.0
        # loss = los1
        if torch.isnan(loss):
            pdb.set_trace()
        return loss   

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

def accuracy(logits_student, target, truth, fn=Thresholding):
    #Related works, fn could be artifact elimination
    #1. detect the filtered sequence
    xs = logits_student
    pred = []
    for x,y in zip(xs, target):
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
            
def measure(logits_student, target, refill, fn=Thresholding):
    with torch.no_grad():
        if torch.is_tensor(logits_student):
            logits_student = logits_student.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        bsflag, noise_ori, batch_snr = refill
        bsflag = bsflag.cpu().numpy()
        noise_ori = noise_ori.cpu().numpy()
        batch_snr = batch_snr.cpu().numpy()
        truth = []
        for z in bsflag:
            if np.sum(z) == 0:
                truth.append( 0 )
            else:
                truth.append( 1 )
        accur = accuracy(logits_student, target, truth, fn=fn)
        corr = Corr(logits_student, target)
        return accur, corr

def measure_val(logits_student, target, refill, fn=Thresholding):
    with torch.no_grad():
        if torch.is_tensor(logits_student):
            logits_student = logits_student.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        bsflag, noise_ori, batch_snr = refill
        bsflag = bsflag.cpu().numpy()
        noise_ori = noise_ori.cpu().numpy()
        batch_snr = batch_snr.cpu().numpy()
        # images = images.cpu().numpy()
        # pdb.set_trace()
        truth = []
        for z in bsflag:
            if np.sum(z) == 0:
                truth.append( 0 )
            else:
                truth.append( 1 )
        accur = accuracy(logits_student, target, truth, fn=fn)
        corr = Corr(logits_student, target)
        impr = SNRDif(logits_student, target, noise_ori, batch_snr, truth)
        return accur, corr, impr

def gen_schheduler(model_student,weight_decay,epochs,learning_rate):
    all_parameters = model_student.parameters()
    weight_parameters = []
    for pname, p in model_student.named_parameters():
        if p.ndimension() == 4 or 'conv' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters)) #id is a fucntion to export the only identifier for each object
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters)) #Other parameters that do not have name
    #Differenciating two type of para. is aimed to apply different optimizing strategies
    optimizer = torch.optim.Adam(
            [{'params' : other_parameters},
            {'params' : weight_parameters, 'weight_decay' : weight_decay}],
            lr=learning_rate,)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/epochs), last_epoch=-1)
    return optimizer, scheduler

def Modify(logits_student, xgg, target, ygg):
    if torch.is_tensor(logits_student):
        logits_student = logits_student.detach().cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    if torch.is_tensor(xgg):
        xgg = xgg.numpy()
    if torch.is_tensor(ygg):
        ygg = ygg.numpy()
    num = logits_student.shape[0]
    assert( num == target.shape[0] )
    pred = []
    truth = []
    for i in range(num):
        ptemp = istft( logits_student[i], xgg[i] )
        ttemp = istft( target[i], ygg[i] )
        pred.append( ptemp )
        truth.append( ttemp )
        # pdb.set_trace()
    return np.array(pred,dtype='float32'), np.array(truth,dtype='float32')
    
def train(epoch, train_loader, model_student, criterion, optimizer, scheduler):
    # pdb.set_trace()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Cor@5', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train() # Only enter into the mode
    end = time.time()

    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
    print('learning_rate:', cur_lr)

    for i, (images, target, wave, mm,nn,kk) in enumerate(train_loader):
        #Input: images, xgg, Output: target, ori_waveform
        refill = (mm,nn,kk)
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)
        wave = wave.to(device)

        # compute outputy
        logits_student = model_student(images)
        # pdb.set_trace()
        loss = criterion(logits_student[0], logits_student[1], target.float(), wave)
        logits_student, target = logits_student[1].detach().cpu().numpy(), wave.cpu().numpy()
        # measure accuracy and record loss
        # prec1 = accuracy(logits_student, target, topk=(1,))
        prec1, corr = measure(logits_student, target, refill)
        n = images.size(0) #denotes the 
        losses.update(loss.item(), n)   #accumulated loss
        # pdb.set_trace()
        top1.update(prec1, n) # the accuracy of a batch and all the samples
        top5.update(corr, n)
        # top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        progress.display(i)

    scheduler.step()
    return losses.avg, top1.avg #, top5.avg

def validate(epoch, val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Cor@5', ':6.2f')
    top9 = MyAverageMeter(ROOT_SNRS, ':6.2f') #Note snrs depend on the import module
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, wave, mm,nn,kk) in enumerate(val_loader):
            #Input: images, xgg, Output: target, ori_waveform
            refill = (mm,nn,kk)
            images = images.to(device)
            target = target.to(device)
            wave = wave.to(device)

            # compute outputy
            logits_student = model(images)
            loss = criterion(logits_student[0], logits_student[1], target.float(), wave)
            logits, target = logits_student[1].detach().cpu().numpy(), wave.cpu().numpy()
            # measure accuracy and record loss
            # pred1 = accuracy(logits, target, topk=(1,))
            prec1, corr, impr = measure_val(logits, target, refill)
            n = images.size(0)
            losses.update(loss.item(), n)
            # pdb.set_trace()
            top1.update(prec1, n)
            top5.update(corr, n)
            top9.update(impr, n)
            # top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # pdb.set_trace()
            progress.display(i)

        x = PrettyTable(field_names=["ROOT_SNRS", "Improv."])
        imean = top9.Avg()
        for key, value in imean.items():
            x.add_row([key,value])
        print(x)
        del x 
        gc.collect()
        print(' * acc@1 {top1.avg:.3f} cor@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
        

    return losses.avg, top1.avg #, top5.avg  
    
         
if __name__ == '__main__':
    #1.1 Setup: GPU, hyperparameters, optimizers and loss
    start_t = time.time()
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    print('This time the {} is used.'.format(device))
    EPOCH = 500
    time_setp = 500
    LR = 0.01 
    batches = 64*3*3
    workers=40
    weight_decay = 0
    criterion = BiLoss()
    
    #1.2 Build the dataset
    a = data_generator(preprocess = False)
    trains = BSdataset_trch(a.trains,preprocess = False)
    train_loader = torch.utils.data.DataLoader(
        trains,
        batch_size=batches, shuffle=True,
        num_workers=workers, pin_memory=True)
    tests = BSdataset_trch(a.tests,preprocess = False)
    val_loader = torch.utils.data.DataLoader(
        tests,
        batch_size=batches, shuffle=False,
        num_workers=workers, pin_memory=True)

    #1.3 Build networks
    # rnn = RNN().to(device)
    
    # rnn = nn.DataParallel(RNN(),device_ids=[0,1,2])
    # rnn.to(device)
    # print(rnn)
    # check = RNN()
    # summary(check, (65,126),device='cpu')
    # input = torch.randn(1, 65, 126)
    # flops, params = profile(check, inputs=(input, ))
    
    # rnn = nn.DataParallel(U_Net(),device_ids=[0,1,2])
    # rnn.to(device)
    # print(rnn)
    # check = U_Net()
    # summary(check, (65,126),device='cpu')
    # input = torch.randn(1, 65, 126)
    # flops, params = profile(check, inputs=(input, ))
    
    rnn = nn.DataParallel(U_NetCom(),device_ids=[0,1,2,3])
    rnn.to(device)
    print(rnn)
    check = U_NetCom()
    summary(check, (4000,),device='cpu')
    input = torch.randn(1, 4000)
    flops, params = profile(check, inputs=(input, ))
    
    print('flops:{}, paras:{}'.format(flops,params))
    # pdb.set_trace()
    optimizer, scheduler = gen_schheduler(rnn, weight_decay, EPOCH, LR)
    
    #2 Training begins
    recording = presentor()
    st = time.time()
    for epoch in range(EPOCH):
        train_obj, train_top1_acc = train(epoch,  train_loader, rnn, criterion, optimizer,scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, rnn, criterion)
        values = {'train/loss':train_obj,'train/accuracy':train_top1_acc,
            'val/loss':valid_obj,'val/accuracy':valid_top1_acc}
        recording.set_values(values,epoch)
        if not ((epoch+1) % 100):
            ssss = 'TimeFreq_'+str(epoch-1)+'.tar'
            dir = os.path.join( os.path.split(os.path.realpath(__file__))[0],'materials', ssss)
            torch.save(rnn,dir)
            print('March to the {}th epoch, a saving is conducted'.format(epoch+1))
    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours'.format(training_time))
    
    #3. Storing
    # dir = os.path.join( os.path.split(os.path.realpath(__file__))[0],'materials','  reuse202101011628.tar' )
    dir = os.path.join( os.path.split(os.path.realpath(__file__))[0],'materials','  pause_u2.tar' )
    # pdb.set_trace()
    torch.save(rnn,dir)
   