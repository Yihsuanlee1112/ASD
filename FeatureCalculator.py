import pandas as pd
import numpy as np
'''
calculate: 計算過程
get: 最後輸出的 DF

Update date: 2023.03.16

'''


class HeadMovement():
    def calculateHeadMovementOnAxis(self, jointTrackingData, axisColumnName):
        distance = 0.
        for i in range(len(jointTrackingData)-1):
            distance += abs(jointTrackingData[axisColumnName]
                            [i+1] - jointTrackingData[axisColumnName][i])

        return distance

    def getHeadAmplitude(self, jointTrackingData, axisColumnName, prefix="", suffix=""):
        jointTrackingData.reset_index(drop=True, inplace=True)
        distance = self.calculateHeadMovementOnAxis(
            jointTrackingData, axisColumnName)
        result = pd.DataFrame([[distance]], columns=[
                              f'{prefix}HeadAmplitude{suffix}'])

        return result


class EyeMovement():
    def calculateEuclideanDistance(self, a, b):
        return np.linalg.norm(a-b)

    def calculateMeanStd(self, targetDataSeries):
        return np.mean(targetDataSeries), np.std(targetDataSeries)

    # def calculateLeftPathLength(self, leftEyeData):
    #     LeftEyePathLength = 0
    #     for i in range(len(leftEyeData)-1):
    #         nowPos = np.array((leftEyeData[f'gaze_origin_mm_X']
    #                           [i], leftEyeData[f'gaze_origin_mm_Y'][i]))
    #         nextPos = np.array((leftEyeData[f'gaze_origin_mm_X']
    #                            [i+1], leftEyeData[f'gaze_origin_mm_Y'][i+1]))
    #         EyePathLength += self.calculateEuclideanDistance(nowPos, nextPos)

    #     return LeftEyePathLength

    def calculatePathLength(self, leftAndRightEyeData):
        EyePathLength = 0
        for i in range(len(leftAndRightEyeData)-1):
            nowPos = np.array((leftAndRightEyeData[f'gaze_origin_mm_X']
                              [i], leftAndRightEyeData[f'gaze_origin_mm_Y'][i]))
            nextPos = np.array((leftAndRightEyeData[f'gaze_origin_mm_X']
                               [i+1], leftAndRightEyeData[f'gaze_origin_mm_Y'][i+1]))
            EyePathLength += self.calculateEuclideanDistance(nowPos, nextPos)

        return EyePathLength

    def getLeftRightEyeFeatures(self, leftAndRightEyeData, prefix="", suffix=""):
        # eyeTypes = ["Left", "Right"]
        leftAndRightEyeData.reset_index(drop=True, inplace=True)

        result = pd.DataFrame()
        for i in range(len(leftAndRightEyeData)-1):
            eyeOpennessMean, eyeOpennessStd = self.calculateMeanStd(
                leftAndRightEyeData[f'eye_openness'])
            eyePathLength = self.calculatePathLength(
                leftAndRightEyeData)

            dfFea = pd.DataFrame([[eyeOpennessMean, eyeOpennessStd, eyePathLength]],
                                 columns=[f"{prefix}EyeOpenness_mean{suffix}", f"{prefix}EyeOpenness_std{suffix}", f'{prefix}EyePathLength{suffix}'])
            result = pd.concat([result, dfFea], axis=1)

        return result

    # def getLeftEyeFeatures(self, leftEyeData, prefix="", suffix=""):

    #     leftEyeData.reset_index(drop=True, inplace=True)

    #     result = pd.DataFrame()
    #     for i in range(len(leftEyeData)-1):
    #         eyeOpennessMean, eyeOpennessStd = self.calculateMeanStd(
    #             leftEyeData[f'eye_openness'])
    #         eyePathLength = self.calculateLeftPathLength(
    #             leftEyeData)

    #     dfFea = pd.DataFrame([[eyeOpennessMean, eyeOpennessStd, eyePathLength]],
    #                          columns=[f"{prefix}LeftEyeOpenness_mean{suffix}", f"{prefix}LeftEyeOpenness_std{suffix}", f'{prefix}LeftEyePathLength{suffix}'])
    #     result = pd.concat([result, dfFea], axis=1)

    #     return result

    # def calculateRightPathLength(self, leftEyeData):
    #     RightEyePathLength = 0
    #     for i in range(len(leftEyeData)-1):
    #         nowPos = np.array((leftEyeData[f'gaze_origin_mm_X']
    #                           [i], leftEyeData[f'gaze_origin_mm_Y'][i]))
    #         nextPos = np.array((leftEyeData[f'gaze_origin_mm_X']
    #                            [i+1], leftEyeData[f'gaze_origin_mm_Y'][i+1]))
    #         EyePathLength += self.calculateEuclideanDistance(nowPos, nextPos)

    #     return RightEyePathLength

    # def getRightEyeFeatures(self, RightEyeData, prefix="", suffix=""):

    #     RightEyeData.reset_index(drop=True, inplace=True)

    #     result = pd.DataFrame()
    #     for i in range(len(RightEyeData)-1):
    #         eyeOpennessMean, eyeOpennessStd = self.calculateMeanStd(
    #             RightEyeData[f'eye_openness'])
    #         eyePathLength = self.calculateRightPathLength(
    #             RightEyeData)

    #     dfFea = pd.DataFrame([[eyeOpennessMean, eyeOpennessStd, eyePathLength]],
    #                          columns=[f"{prefix}RightEyeOpenness_mean{suffix}", f"{prefix}RightEyeOpenness_std{suffix}", f'{prefix}RightEyePathLength{suffix}'])
    #     result = pd.concat([result, dfFea], axis=1)

    #     return result

    def getFocusRate(self, eyeFocusdata, targetName, prefix="", suffix=""):
        eyeFocusdata.reset_index(drop=True, inplace=True)
        num_target = sum(
            (eyeFocusdata['FocusObjectName'] == targetName).astype(int))
        num_all = len(eyeFocusdata)
        try:
            rate = num_target / num_all
        except:
            rate = np.nan
        result = pd.DataFrame([[rate]], columns=[f'{prefix}FocusRate{suffix}'])

        return result


class EEG():
    def __init__(self, PICO_CHANNELS_ORDER) -> None:
        self.SAMPLING_RATE = 200
        self.FMIN = 2.
        self.FMAX = 30.
        self.all_ch = ['ch1', 'ch2', 'ch3', 'ch4']
        # PICO_CHANNELS_ORDER 與 Ganglion 的接線有關係，分別對照 ['ch1', 'ch2', 'ch3', 'ch4']
        self.PICO_CHANNELS_ORDER = PICO_CHANNELS_ORDER

        self.delta_band = [0.5, 4]
        self.theta_band = [4, 8]
        self.alpha_band = [8, 12]
        self.beta_band = [12, 30]
        self.gamma_band = [30, 100]

        return

    def filterEeg(self, eegData):
        '''
        Filter eeg data, eegData must be float64
        '''
        from mne import filter
        for ch in self.all_ch:
            eegData[ch] = filter.filter_data(eegData[ch].to_numpy().astype(
                np.float64), sfreq=self.SAMPLING_RATE, l_freq=self.FMIN, h_freq=self.FMAX, verbose=False)
            eegData[ch] = filter.notch_filter(eegData[ch].to_numpy().astype(np.float64), Fs=self.SAMPLING_RATE, freqs=np.arange(
                60, 100, 60), verbose=False, filter_length='auto', phase='zero')
        return eegData

    def createColumns(self):
        eegBands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        cols = []
        for ch in self.PICO_CHANNELS_ORDER:
            for eegBand in eegBands:
                cols.append(f"{ch}_{eegBand}")

        return cols

    def bandpower(self, data, sf, band, window_sec=None, relative=False):
        """Compute the average power of the signal x in a specific frequency band.
        https://raphaelvallat.com/bandpower.html?utm_source=pocket_mylist
        Requires MNE-Python >= 0.14.

        Parameters
        ----------
        data : 1d-array
        Input signal in the time-domain.
        sf : float
        Sampling frequency of the data.
        band : list
        Lower and upper frequencies of the band of interest.
        method : string
        Periodogram method: 'welch' or 'multitaper'
        window_sec : float
        Length of each window in seconds. Useful only if method == 'welch'.
        If None, window_sec = (1 / min(band)) * 2.
        relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

        Return
        ------
        bp : float
        Absolute or relative band power.
        """
        from scipy.integrate import simps
        from scipy import signal

        band = np.asarray(band)
        low, high = band

        # sd, freqs = psd_array_multitaper(data, sf, adaptive=True, normalization='full', verbose=0)
        freqs, psd = signal.welch(data, sf, nperseg=window_sec * sf)

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]

        # Find index of band in frequency vector
        idx_band = np.logical_and(freqs >= low, freqs <= high)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        bp = simps(psd[idx_band], dx=freq_res)

        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp

    def getBandPower(self, eegData, prefix="", suffix=""):
        result = pd.DataFrame(columns=self.createColumns())

        eegData = self.filterEeg(eegData)

        for ch, ach in zip(self.all_ch, self.PICO_CHANNELS_ORDER):
            delta_bandpower = self.bandpower(eegData[ch], self.SAMPLING_RATE, self.delta_band, window_sec=len(
                eegData[ch])/self.SAMPLING_RATE, relative=True)
            theta_bandpower = self.bandpower(eegData[ch], self.SAMPLING_RATE, self.theta_band, window_sec=len(
                eegData[ch])/self.SAMPLING_RATE, relative=True)
            alpha_bandpower = self.bandpower(eegData[ch], self.SAMPLING_RATE, self.alpha_band, window_sec=len(
                eegData[ch])/self.SAMPLING_RATE, relative=True)
            beta_bandpower = self.bandpower(eegData[ch], self.SAMPLING_RATE, self.beta_band, window_sec=len(
                eegData[ch])/self.SAMPLING_RATE, relative=True)
            gamma_bandpower = self.bandpower(eegData[ch], self.SAMPLING_RATE, self.gamma_band, window_sec=len(
                eegData[ch])/self.SAMPLING_RATE, relative=True)

            result[f'{prefix}{ach}{suffix}_delta'] = [delta_bandpower]
            result[f'{prefix}{ach}{suffix}_theta'] = [theta_bandpower]
            result[f'{prefix}{ach}{suffix}_alpha'] = [alpha_bandpower]
            result[f'{prefix}{ach}{suffix}_beta'] = [beta_bandpower]
            result[f'{prefix}{ach}{suffix}_gamma'] = [gamma_bandpower]

        return result


class ADHDTaskPerformance():
    def divide(self, x, y):
        try:
            return x/y
        except ZeroDivisionError:
            return -1

    def calculateConfusionMatrix(self, taskPerformanceData):
        taskPerformanceData.reset_index(drop=True, inplace=True)
        try:
            TP = sum((taskPerformanceData["Answer"] == True) & (
                taskPerformanceData["IsCorrect"] == True).astype(int))
            FP = sum((taskPerformanceData["Answer"] == False) & (
                taskPerformanceData["IsCorrect"] == False).astype(int))
            FN = sum((taskPerformanceData["Answer"] == True) & (
                taskPerformanceData["IsCorrect"] == False).astype(int))
            TN = sum((taskPerformanceData["Answer"] == False) & (
                taskPerformanceData["IsCorrect"] == True).astype(int))
        except:  # WCST 沒有 Answer 欄位
            TP = sum((taskPerformanceData["IsCorrect"] == True).astype(int))
            FP = sum((taskPerformanceData["IsCorrect"] == False).astype(int))
            FN = sum((taskPerformanceData["IsCorrect"] == False).astype(int))
            TN = sum((taskPerformanceData["IsCorrect"] == True).astype(int))

        return TP, FP, FN, TN

    def getTaskAccuracy(self, taskPerformanceData, prefix="", suffix=""):
        TP, FP, FN, TN = self.calculateConfusionMatrix(taskPerformanceData)
        # Accuracy
        accuracy = self.divide(TP+TN, TP+FP+FN+TN)
        # omissionError
        omissionError = self.divide(FN, FN+TP)
        # commissionError: inappropriate response to the nontarget stimulus
        commissionError = self.divide(FP, FP+TN)

        result = pd.DataFrame([[accuracy, omissionError, commissionError]],
                              columns=[f"{prefix}Accuracy{suffix}", f"{prefix}OmissionError{suffix}", f"{prefix}CommissionError{suffix}"])

        return result

    def getReactiveTime(self, taskPerformanceData, prefix="", suffix=""):
        maxCurrentWaitTime = taskPerformanceData['CurrentWaitTime'].max()

        col_reactive = taskPerformanceData["ReactiveTime"][taskPerformanceData.ReactiveTime != maxCurrentWaitTime]
        if len(col_reactive) < 1:
            reactiveTime = maxCurrentWaitTime
        else:
            reactiveTime = np.mean(col_reactive)
        result = pd.DataFrame([[reactiveTime]], columns=[
                              f"{prefix}ReactiveTime{suffix}"])

        return result
