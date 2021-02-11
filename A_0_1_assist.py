import os
from feature_transformation import read_wav_data, GetFrequencyFeatures 
import matplotlib.pyplot as plt
if __name__ == '__main__':
    apath = '../../Data/BSfilteringData/data/762758/bowels/81730697302190_2018_12_20_12_02_42.wav'
    wavsignal, fs = read_wav_data(apath)
    wavsignal = wavsignal[0]
    plt.plot(wavsignal)
    plt.show()