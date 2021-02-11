import tgt
def get_filelist(dir,Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(newDir)
        # pdb.set_trace()
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # print(dir)
            newDir=os.path.join(dir,s)
            # print(newDir)
            # pdb.set_trace()
            get_filelist(newDir,Filelist)
    else:
        print('"{}" is neither a file, not a folder!'.format(dir))
    return Filelist

def read_in_single_textgrid(file_name):
    #file_name = os.path.join('/home/zx/Dolphin/Data/textgrid/test/20191011/zcz/1616893', '37429442330632_2019_09_18_01_45_11.TextGrid')
    tg_data = tgt.read_textgrid(file_name) # read a Praat TextGrid file and return a TextGrid object
#    print(tg_data)
    tier_names = tg_data.get_tier_names()  # get names of all tiers
#    print (tier_names)
    return tg_data

def extract_interval(obj):
    a = []
    for i in range(len(obj)):
        a.append( (obj[i].start_time, obj[i].end_time, obj[i].text) )
    return a

def marking(file_path):
    ddir, filename = os.path.split(file_path)
    pre = ''
    if ('fast' in ddir) and ('20190525' in ddir):
        pre = 'fas_0525_'+ddir[-1-1:]
    elif ('fast' in ddir) and ('20190526' in ddir):
        pre = 'fas_0526_'+ddir[-1-1:]
    elif ('fed' in ddir) and ('20190525half0930to1000' in ddir):
        pre = 'fed_0525_'+ddir[-1-1:]
    else:
        assert(0)
    idx = pre + '_' + filename.replace('.','_')
    return idx

if __name__=="__main__":
    lll = {}
    src = os.path.join('E:\\2020Great','liujuzheng','liujuzhengdata')
    src1 = os.path.join(src,'fast','20190525')
    src2 = os.path.join(src,'fast','20190526')
    src3 = os.path.join(src,'fed','20190525half0930to1000')
    srcs = [src1,src2,src3]
    # print(srcs[0])
    onfire = []
    for branch in srcs:
        temp = []
        temp = get_filelist(branch,temp)
        # print(len(temp))
        onfire = onfire + temp
    tgfiles = [i for i in onfire if 'TextGrid' in i]
    # print(tgfiles[1:10])
    # pdb.set_trace()
    print(len(tgfiles))
    for i in tgfiles:
        # print(i)
        tgob = read_in_single_textgrid(i)
        assert(len(tgob.tiers)==1)
        info = extract_interval(tgob.tiers[0].intervals)
        idx = marking(i)
        # pdb.set_trace()
        lll[idx]=(info,i)
    # export to 