import wave
import numpy as np
from scipy.fftpack import fft
from python_speech_features import mfcc
from librosa.core import istft as librosa_istft
from librosa.core import stft as librosa_stft
import matplotlib.pyplot as plt
import pdb
import torch
eps = 1e-5

def griding_new(datum, row_number, colum_edge, normal_features=False):
    assert(len(datum)%row_number==0)
    assert(colum_edge[-1]>colum_edge[0])
    datum = np.array(datum)
    geshu = len(datum)//row_number
    feature = np.zeros((len(colum_edge)-1,row_number),dtype=np.uint8)
    for i in range(row_number):
        seg = datum[i*geshu:(i+1)*geshu]
        alist , _ = np.histogram(seg,bins=colum_edge)
        feature[:,i] = alist[::-1]
    if not normal_features:
        temp = feature[20:32,:]
        det = [True]*5+[False]*2+[True]*5
        return temp[det,:]
    else:
        return feature[20:32,:]

def stft(x, **kwargs):
    """
    Only for 1xL tensors, i.e. C = 1
    https://librosa.github.io/librosa/generated/librosa.core.stft.html#librosa-core-stft
    
    S_abs is the magnitude of frequency bin f at frame t
    
    The integers t and f can be converted to physical units by means of
    the utility functions frames_to_sample and fft_frequencies.
    
    """
    if torch.is_tensor(x):
        S = librosa_stft(x.squeeze().cpu().numpy(), **kwargs)
    else:
        S = librosa_stft(np.array(x,dtype='float32'), **kwargs)
    # S_abs = torch.tensor(np.abs(S), dtype=torch.double,
                         # device=DEVICE).unsqueeze(dim=0)
    # S_ang = torch.tensor(np.angle(S), dtype=torch.double,
                         # device=DEVICE).unsqueeze(dim=0)
    S_abs = np.abs(S)
    S_ang = np.angle(S)
    
    return S_abs, S_ang


def istft(S_module, S_angle, length=None, **kwargs):
    """
    Only for 1xL tensors, i.e. C = 1
    https://librosa.github.io/librosa/generated/librosa.core.istft.html#librosa.core.istft
    """
    if torch.is_tensor(S_module) or torch.is_tensor(S_angle): 
        S_module = S_module.squeeze().cpu().numpy()
        S_angle = S_angle.squeeze().cpu().numpy()
    else:
        S_module = np.array(S_module)
        S_angle = np.array(S_angle)
    S_complex = S_module * np.exp(1j * S_angle)
    y = librosa_istft(S_complex, length=length, **kwargs)
    # return torch.as_tensor(y, device=DEVICE)
    return y

def read_wav_data(filename):
	'''
	读取一个wav文件，返回声音信号的时域谱矩阵和播放时间
	'''
	wav = wave.open(filename,"rb") # 打开一个wav格式的声音文件流
	num_frame = wav.getnframes() # 获取帧数
	num_channel=wav.getnchannels() # 获取声道数
	framerate=wav.getframerate() # 获取帧速率
	num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
	str_data = wav.readframes(num_frame) # 读取全部的帧
	wav.close() # 关闭流
	wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
	wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
	wave_data = wave_data.T # 将矩阵转置
	#wave_data = wave_data
	return wave_data, framerate

def GetFrequencyFeatures(wavsignal, fs, feature_dimension = 200,frame_length = 400, shift=400):

	length = frame_length
	nfft = int(feature_dimension*2)
	#1. forming the time-window
	x = np.linspace(0, length - 1, length, dtype=np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (length - 1))  # 汉明窗

	wav_arr = np.array(wavsignal)
	#Newly added
	tmp = wav_arr.shape[0]
	wav_arr = wav_arr.reshape(1,tmp)
	wav_length = wav_arr.shape[1]

	range0_end = (wav_length - length) // shift + 1
	data_input = np.zeros((range0_end, nfft//2), dtype=np.float)  # 用于存放最终的频率特征数据
	if np.sum(wavsignal) == 0:
		return data_input
	else:
		data_line = np.zeros((1, nfft), dtype=np.float)

		for i in range(0, range0_end):
			p_start = i * shift
			p_end = p_start + length
			data_line = wav_arr[0, p_start:p_end]
			data_line = data_line * w  # 加窗

			data_line = np.abs(fft(data_line,n=nfft)) / wav_length
			data_input[i] = data_line[0:nfft//2]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的

		# print(data_input.shape)
		data_input = np.log(data_input+eps)
		data_input = (data_input - np.min(data_input)) / np.ptp(data_input)
		# data_input = (data_input - data_input.mean()) / data_input.std()
		return data_input

def SimpleMfccFeatures(wave_data, samplerate, shift=0.1, featurelength=26):
    temp = mfcc(wave_data[0], samplerate=samplerate, winlen=0.1, winstep=shift, numcep=featurelength, appendEnergy=False)
    # return stats.zscore(temp)
    b = (temp - np.min(temp)) / np.ptp(temp)
    return b

def SimplifiedMfccFeatures(wave_data, samplerate=4000, shift=0.1, featurelength=26):
    # pdb.set_trace()
    wave_data = np.array(wave_data,dtype='float32')
    temp = mfcc(wave_data, samplerate=samplerate, winlen=0.1, winstep=shift, numcep=featurelength, appendEnergy=False)
    # return stats.zscore(temp)
    b = (temp - np.min(temp)) / np.ptp(temp)
    return b


from math import floor
def myround(a):
	b=floor(a)
	if (a-b)>=0.5:
		return (b+1)
	else:
		return b

if __name__ == '__main__':
    curve = np.random.randn(4000)
    am,ph = stft(curve, n_fft=128)
    recov = istft(am,ph)
    plt.subplot(211)
    plt.plot(curve)
    plt.subplot(212)
    plt.plot(recov)
    plt.show()
    pdb.set_trace()
# if __name__ == '__main__':
	# a = 2.5
	# print(myround(a))
	# b=5.6
	# print(myround(b))
	# c=25946313325.4566531113333
	# print(myround(c))