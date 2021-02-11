import os
import tgt
import pdb
from snr_computing import read_in_single_textgrid, extract_interval
# excludes = ('526006','526007','526008','526009','20190526.zip')
excludes = ('20190526.zip')
# dirpath = 'I:\thudata\all_filtered_data_&&_alittle_routines\filtered\fast\20190526'
# dirpath = os.path.join('I:\\','thudata','all_filtered_data_&&_alittle_routines','filtered','fast','20190526')
dirpath = os.path.join('I:\\','thudata','all_filtered_data_&&_alittle_routines','filtered','fed','20190525')
# print(dirpath)
rest = [ (os.path.join(dirpath,i),i) for i in os.listdir(dirpath) if i not in excludes]
# print(rest)
statistic = []
for apath, afolder in rest:
    files =[os.path.join(apath,j) for j in os.listdir(apath) if 'TextGrid' in j]
    tmpcount = []
    for tgfile in files:
        tgdata = read_in_single_textgrid(tgfile)
        info = extract_interval(tgdata.tiers[0].intervals)
        # print(info)
        tc = len([x for x in info if x[2] == 'T'])
        tmpcount.append(tc)
        # print(info)
        # pdb.set_trace()
    psoncount = sum(tmpcount)
    statistic.append((afolder,psoncount))

print(statistic)