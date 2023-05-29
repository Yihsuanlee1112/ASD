import numpy as np
from scipy.signal import convolve
import pandas as pd
import os

class Morlet:
    def __init__(self) -> None:

        return

    def MorletWavelet(self, fc):
        # MorletWavelet: Morlet wavelet.
        # [MW] = MorletWavelet(fc) returns coefficients of 
        # the Morlet wavelet, where fc is the central frequency.

        # MW(fc,t) = A * exp(-t^2/(2*sigma_t^2)) * exp(2i*PI*fc*t)
        #            = A * exp(t*(t/(-2*sigma_t^2)+2i*PI*fc))
        # A = 1/(sigma_t*PI^0.5)^0.5,
        # sigma_t = 1/(2*PI*sigma_f)
        # sigma_f = fc/F_RATIO
        #
        # The effective support of this wavelet is determined by fc and two constants,
        # fc/sigma_f and Z_alpha/2.
        #
        # Yong-Sheng Chen

        # Compute values of the Morlet wavelet.

        F_RATIO = 8    # frequency ratio (number of cycles): fc/sigma_f, should be greater than 5
        Zalpha2 = 3.3  # value of Z_alpha/2, when alpha=0.001

        sigma_f = fc/F_RATIO
        sigma_t = 1/(2*np.pi*sigma_f)
        A = 1/np.sqrt(sigma_t*np.sqrt(np.pi))
        max_t = np.ceil(Zalpha2*sigma_t)

        t = np.arange(-max_t, max_t+1)

        v1 = 1/(-2*sigma_t**2)
        v2 = 2j*np.pi*fc
        MW = A * np.exp(t*(t*v1+v2))

        return MW


    def tfa_morlet(self, td, fs, fmin, fmax, fstep):
        # Initialize output variables
        OUTf = []
        TF = []
        nco_len = []

        for fc in np.arange(fmin, fmax+fstep, fstep):
            MW = self.MorletWavelet(fc/fs)   # calculate the Morlet Wavelet by giving the central freqency
            MWHL = (len(MW)-1)//2       # half length of the Morlet Wavelet
            nco_len.append(len(MW))
            
            cr = convolve(td, MW, mode='same')    
            cr = cr[MWHL:len(cr)-MWHL]
            TF = np.hstack([TF, cr.reshape(-1, 1)])

        OUTf = np.abs(TF)

        return OUTf, nco_len
    
class EEGPreprocessor:
    def __init__(self, PICO_CHANNELS_ORDER) -> None:
        self.PICO_CHANNELS_ORDER = PICO_CHANNELS_ORDER

        return       
    # 怎麼把TP位進來?
    def getEventTimes(self, task, path):
        if task == "C":
            fileFullPath = path + 'ChildClassCptTestData.csv'
        elif task == "A":
            fileFullPath = path + 'ChildClassAudioTestData.csv'
        else:
            return None
        
        if not os.path.isfile(fileFullPath):
            print("[getEventTimes] no test data file found in folder")
            return None
        
        opts = pd.read_csv(fileFullPath, nrows=0).columns
        opts = {col: str for col in opts}
        opts['usecols'] = ['Base_Word', 'Base_Timestamp', 'Base_Sequence']
        gameRecord = pd.read_csv(fileFullPath, dtype=opts)
        
        gameRecord.dropna(inplace=True)
        timeSequence = transFormatTimestampToSec(gameRecord['Base_Timestamp'])
        infSequence = gameRecord['Base_Sequence'].tolist()
        wordSequence = gameRecord['Base_Word'].tolist()

        time = []
        isGo = []
        isInf = []
        totalLength = len(wordSequence)
        for i in range(totalLength):
            currentWord = wordSequence[i]
            if (currentWord == 0 or currentWord == 'A') and i != totalLength-1:
                time.append(timeSequence[i+1])
                isGo.append(wordSequence[i+1] == 1 or wordSequence[i+1] == 'X')
                isInf.append(infSequence[i+1] != 'None')

        eventInfoTable = pd.DataFrame({'time': time, 'isGo': isGo, 'isInf': isInf})
        return eventInfoTable
    
    def EEGPreprocess(self, data):      
        
        timeSeqWithLabal = getEventTimes(task, path)

        [EventDataList, eventClosestTimeList] = getEventData(data, timeSeqWithLabal);

        if size(EventDataList, 3) ~= length(timeSeqWithLabal.isGo)
            disp("Cannot calculate the features due to missing values")
            return;
        end
        saveEventData(EventDataList, dataName, outputPath, dataType);

    


%% Get 4 kind induce band data
    goInfFreqData = extractInduceFrequencyBand(timeSeqWithLabal, EventDataList, true, true);
    dataType = 'goInfInduceData';
    saveEventData(goInfFreqData, dataName, outputPath, dataType);

    goNoinfFreqData = extractInduceFrequencyBand(timeSeqWithLabal, EventDataList, true, false);
    dataType = 'goNoinfInduceData';
    saveEventData(goNoinfFreqData, dataName, outputPath, dataType);

    nogoInfFreqData = extractInduceFrequencyBand(timeSeqWithLabal, EventDataList, false, true);
    dataType = 'nogoInfInduceData';
    saveEventData(nogoInfFreqData, dataName, outputPath, dataType);

    nogoNoinfFreqData = extractInduceFrequencyBand(timeSeqWithLabal, EventDataList, false, false);
    dataType = 'nogoNoinfInduceData';
    saveEventData(nogoNoinfFreqData, dataName, outputPath, dataType);

    [~, goInfInduceFrequencyFeature] = getFeatures([], goInfFreqData);
    [~, goNoinfInduceFrequencyFeature] = getFeatures([], goNoinfFreqData);
    [~, nogoInfInduceFrequencyFeature] = getFeatures([], nogoInfFreqData);
    [~, nogoNoinfInduceFrequencyFeature] = getFeatures([], nogoNoinfFreqData);
    
    induceFeature = [goInfInduceFrequencyFeature, goNoinfInduceFrequencyFeature, nogoInfInduceFrequencyFeature, nogoNoinfInduceFrequencyFeature];
    

%% Get 4 kind evoke band data and mean time data
    [goInfFreqData, goInfMeanData] = extractEvokeFrequencyBand(timeSeqWithLabal, EventDataList, true, true);
    dataType = 'goInfEvokeData';
    %disp(dataType)
    saveEventData(goInfFreqData, dataName, outputPath, dataType);
    dataType = 'goInfMeanData';
    saveEventData(goInfMeanData, dataName, outputPath, dataType);

    [goNoinfFreqData, goNoinfMeanData] = extractEvokeFrequencyBand(timeSeqWithLabal, EventDataList, true, false);
    dataType = 'goNoinfEvokeData';
    %disp(dataType)
    saveEventData(goNoinfFreqData, dataName, outputPath, dataType);
    dataType = 'goNoinfMeanData';
    saveEventData(goNoinfMeanData, dataName, outputPath, dataType);

    [nogoInfFreqData, nogoInfMeanData] = extractEvokeFrequencyBand(timeSeqWithLabal, EventDataList, false, true);
    dataType = 'nogoInfEvokeData';
    %disp(dataType)
    saveEventData(nogoInfFreqData, dataName, outputPath, dataType);
    dataType = 'nogoInfMeanData';
    saveEventData(nogoInfMeanData, dataName, outputPath, dataType);

    [nogoNoinfFreqData, nogoNoinfMeanData] = extractEvokeFrequencyBand(timeSeqWithLabal, EventDataList, false, false);
    dataType = 'nogoNoinfEvokeData';
    %disp(dataType)
    saveEventData(nogoNoinfFreqData, dataName, outputPath, dataType);
    dataType = 'nogoNoinfMeanData';
    saveEventData(nogoNoinfMeanData, dataName, outputPath, dataType);

    [goInfTimeFeature, goInfEvokeFrequencyFeature] = getFeatures(goInfMeanData, goInfFreqData);
    [goNoinfTimeFeature, goNoinfEvokeFrequencyFeature] = getFeatures(goNoinfMeanData, goNoinfFreqData);
    [nogoInfTimeFeature, nogoInfEvokeFrequencyFeature] = getFeatures(nogoInfMeanData, nogoInfFreqData);
    [nogoNoInfTimeFeature, nogoNoinfEvokeFrequencyFeature] = getFeatures(nogoNoinfMeanData, nogoNoinfFreqData);
    
    evokeFeature = [goInfEvokeFrequencyFeature, goNoinfEvokeFrequencyFeature, nogoInfEvokeFrequencyFeature, nogoNoinfEvokeFrequencyFeature];
    
%% save time and frequency features
    frequencyFeature = [induceFeature evokeFeature];
    dataType = 'frequencyFeature';
    saveData(frequencyFeature, dataName, outputPath, dataType);
    
    dataType = 'timeFeature';
    timeFeature = [goInfTimeFeature goNoinfTimeFeature nogoInfTimeFeature nogoNoInfTimeFeature];
    saveData(timeFeature, dataName, outputPath, dataType);

end

function saveData(data, name, outputPath, dataType)
    outputFolder = strcat(outputPath, '/', dataType, '/');
    if ~exist(outputFolder, 'dir')
       mkdir(outputFolder)
    end

    path = strcat(outputFolder, name, '.csv');
    switch(class(data))
        case 'table'
            writetable(data, path);
        otherwise
            writematrix(data, path);
    end
end