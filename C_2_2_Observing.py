import os
import numpy
from C_2_1_Freq_synthesis import U_Net,Modify,measure_val,conv_block,up_conv
from A_2_0_Freq_OpGeData import ROOT_SNRS, data_generator, BSdataset_recovery
from feature_transformation import istft 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
from Denoise_BS.utils.utils import *
from prettytable import PrettyTable
import gc
import pdb

def setup_model(model_teacher):
    for p in model_teacher.parameters():
        p.requires_grad = False
    model_teacher.eval()
    return model_teacher

def unfold(whole_package):
    new_package = []
    for geti in whole_package:
        new_geti = []
        for tile in geti:
            if torch.is_tensor(tile):
                tile = tile.cpu().numpy()
            new_geti.append( tile )
        # pdb.set_trace()
        assert(len(new_geti) == 7)
        images, xgg, pred, truth, mm,nn,kk = new_geti
        assert( pred.shape[0] == truth.shape[0] )
        assert( truth.shape[0] == mm.shape[0] )
        assert( mm.shape[0] == nn.shape[0] )
        assert( nn.shape[0] == kk.shape[0] )
        temp = []
        for i in range(pred.shape[0]):
            ori_wav = istft( images[i], xgg[i] )
            temp.append( ( ori_wav, pred[i], truth[i], mm[i], nn[i], kk[i] ) )
        new_package = new_package + temp
    return new_package

def validate(epoch, val_loader, model, criterion, memory = True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Cor@5', ':6.2f')
    top9 = MyAverageMeter(ROOT_SNRS, ':6.2f') #Note snrs depend on the import module
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    
    if memory:
        stuff = []
    
    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target, mm,nn,kk,xgg,ygg) in enumerate(val_loader):
            refill = (mm,nn,kk)
            images = images.to(device)
            target = target.to(device)

            # compute output
            logits = model(images)
            loss = criterion(logits, target.float())
            logits, target = Modify(logits, xgg, target, ygg)
            
            if memory:
                stuff.append( (images, xgg ,logits, target, mm, nn, kk ) )
            
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
        
        

    return losses.avg, top1.avg, stuff #, top5.avg  
    
    
    
if __name__ == '__main__':
    #1. trivial
    start_t = time.time()
    batches = 64*3*3
    workers=40
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")
    print('This time the {} is used.'.format(device))
    #2 Build the dataset
    a = data_generator(preprocess = False)
    trains = BSdataset_recovery(a.trains,preprocess = False)
    train_loader = torch.utils.data.DataLoader(
        trains,
        batch_size=batches, shuffle=True,
        num_workers=workers, pin_memory=True)
    tests = BSdataset_recovery(a.tests,preprocess = False)
    val_loader = torch.utils.data.DataLoader(
        tests,
        batch_size=batches, shuffle=False,
        num_workers=workers, pin_memory=True)
    #3. networks
    # rnn = U_Net()
    rnn = nn.DataParallel(U_Net(),device_ids=[0,1,2,3])
    # rnn = rnn.module if hasattr(rnn, 'module') else rnn
    rnn.to(device)
    dir = os.path.join( os.path.split(os.path.realpath(__file__))[0],'materials','  pause.tar' )
    temp = torch.load(dir)
    rnn = temp
    # pdb.set_trace()
    # rnn.load_state_dict( temp )
    rnn = setup_model(rnn)
    
    criterion = nn.MSELoss()
    epoch = 0
    valid_obj, valid_top1_acc, pred_package = validate(epoch, val_loader, rnn, criterion)
    print('val loss:',valid_obj)
    print('val accuracy:',valid_top1_acc)
    cleared_pack = unfold(pred_package)
    i = 0
    # Store for the next stage
    import pickle
    _, _, pred_train_package = validate(epoch, train_loader, rnn, criterion)
    cleared_train_pack = unfold(pred_train_package)
    opa = os.path.join( os.getcwd(), 'materials','FirstStageData.pkl' )
    with open(opa, 'wb') as f:
        pickle.dump((cleared_train_pack,cleared_pack), f)
        print('Data of the first stage dumpped.')
    pdb.set_trace()
    for wav_ori, pred, truth, bsflag, noise_ori, snr in cleared_pack:
        if snr == ROOT_SNRS[i]:
            i = i + 1
            if i == len(ROOT_SNRS):
                i = 0
            if np.sum(bsflag) == 0:
                label = 'NBS_' + str(snr)
            else:
                label = 'BS_' + str(snr)
            ax = plt.subplot(411)
            ax.plot(wav_ori)
            ax.set_ylabel('wav_ori')
            ax = plt.subplot(412)
            ax.plot(pred)
            ax.set_ylabel('prediction')
            ax = plt.subplot(413)
            ax.plot(truth)
            ax.set_ylabel('truth')
            ax = plt.subplot(414)
            ax.plot(noise_ori)
            ax.set_ylabel('noise_ori')
            plt.suptitle(label)
            plt.show()
            pdb.set_trace()