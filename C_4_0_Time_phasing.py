import os
import numpy as np
import pickle
import torch
import time
import sys
sys.path.append('..')
from Denoise_BS.utils.visuals import presentor
from torch.utils.data import Dataset
from thop import profile
from prettytable import PrettyTable
from math import log10 as log
import matplotlib.pyplot as plt
import pdb
from C_1_0_NNCounterpart import *


class Thresholding(nn.Module):
    def __init__(self,size=4000):
        super(Thresholding, self).__init__()
        
        self.lambda_1_m1 = nn.Parameter(torch.ones(size)*(-1000), requires_grad=True)

    def forward(self, x):
        mask = (torch.abs(x)+self.lambda_1_m1>0)
        x = x * mask
        # pdb.set_trace()
        # x[torch.abs(x)+self.lambda_1_m1<0] = 0
        return x
        
    def getitem():
        print(self.lambda_1_m1)


class Thresholding2(nn.Module):
    def __init__(self,size=4000):
        super(Thresholding2, self).__init__()
        
        self.lambda_1_m1 = nn.Parameter(torch.ones(size), requires_grad=True)
        self.lambda_2_m1 = nn.Parameter(torch.zeros(size), requires_grad=True)

    def forward(self, x):
        # mask = (torch.abs(x)+self.lambda_1_m1>0)
        # x = x * mask
        x = x * self.lambda_1_m1 + 10 * self.lambda_2_m1
        # pdb.set_trace()
        # x[torch.abs(x)+self.lambda_1_m1<0] = 0
        return x
        
    def getitem(self,):
        print(self.lambda_1_m1)
        print(self.lambda_2_m1)

#Simple model
class RNN(nn.Module):
    def __init__(self,input_size=1):
        super(RNN, self).__init__()
        print('It is a LSTM!')
        self.lstm = nn.GRU(  # if use nn.RNN(), it hardly learns
            input_size=input_size,
            hidden_size=64,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            # batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            # bidirectional=True,
        )
        # self.out = nn.Linear(4000, 4000)
        self.out = Thresholding2()
    def forward(self, x): #x.size()； [batch_size, time_steps, features]
        # x = x.reshape( x.shape[0], 4000, 1 )
        # r_out, h_c = self.lstm(x, None)
        # out = self.out(r_out)
        out = self.out(x)
        # out = torch.squeeze( out )
        return out
        
    def getitem(self,):
        self.out.getitem()

#Dataset
class BSdataset_FirstStage(Dataset):
    def __init__(self, mydata, transform=None, preprocess = True):
        self.data = mydata
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data[idx][1]
        y = self.data[idx][2]
        x = np.array(x, dtype='float32')
        # z = self.data[idx][2]
        bs_flag = self.data[idx][3]
        noise_ori = self.data[idx][4]
        snr = self.data[idx][5] 
        if self.preprocess:
            x = x.reshape(1,x.shape[0],x.shape[1])
        else:
            pass
            # x = x.reshape(x.shape[0],1)
        if self.transform is not None:
            x = self.transform(x)
            x = np.array(x)
        y = np.array(y, dtype='float32')
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(bs_flag), torch.tensor(noise_ori), torch.tensor(snr)

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

    for i, (images, target, mm,nn,kk) in enumerate(train_loader):
        refill = (mm,nn,kk)
        data_time.update(time.time() - end)
        images = images.to(device)
        target = target.to(device)

        # compute outputy
        logits_student = model_student(images)
        # logits_teacher = model_teacher(images)
        # pdb.set_trace()
        loss = criterion(logits_student, target.float())
        # loss.requires_grad = True
        # measure accuracy and record loss
        # prec1 = accuracy(logits_student, target, topk=(1,))
        # prec1, corr = measure(logits_student, target, refill)
        prec1=0 
        corr=0
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
        for i, (images, target, mm,nn,kk) in enumerate(val_loader):
            refill = (mm,nn,kk)
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            loss = criterion(logits, target.float())

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
    #0. Trivial
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
    criterion = nn.MSELoss()
    #1. Load the data of the first stage
    npath = os.path.join( os.getcwd(), 'materials','FirstStageData.pkl' )
    with open(npath, 'rb') as n:
        data = pickle.load(n)
        print('A dataset is loaded.')
    trains = BSdataset_FirstStage(data[0],preprocess = False)
    train_loader = torch.utils.data.DataLoader(
        trains,
        batch_size=batches, shuffle=True,
        num_workers=workers, pin_memory=True)
    tests = BSdataset_FirstStage(data[1],preprocess = False)
    val_loader = torch.utils.data.DataLoader(
        tests,
        batch_size=batches, shuffle=False,
        num_workers=workers, pin_memory=True)
    #2. construct the model
    rnn = RNN().to(device)
    print(rnn)
    check = RNN()
    # summary(check, (4000,),device='cpu')
    input = torch.randn(1, 4000)
    flops, params = profile(check, inputs=(input, ))
    print('flops:{}, paras:{}'.format(flops,params))
    
    optimizer, scheduler = gen_schheduler(rnn, weight_decay, EPOCH, LR)
    #3. training
    recording = presentor()
    st = time.time()
    for epoch in range(EPOCH):
        train_obj, train_top1_acc = train(epoch,  train_loader, rnn, criterion, optimizer,scheduler)
        valid_obj, valid_top1_acc = validate(epoch, val_loader, rnn, criterion)
        rnn.getitem()
        values = {'train/loss':train_obj,'train/accuracy':train_top1_acc,
            'val/loss':valid_obj,'val/accuracy':valid_top1_acc}
        recording.set_values(values,epoch)
    training_time = (time.time() - start_t) / 3600
    print('total training time = {} hours'.format(training_time))