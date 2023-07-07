import os
import re
import sys
import librosa
from random import shuffle
import numpy as np
from typing import Tuple, Union
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import utils

def features(X, sample_rate: float) -> np.ndarray:
    stft = np.abs(librosa.stft(X))

    # fmin 和 fmax 对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(y=X, sr=sample_rate, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # 频谱质心
    cent = librosa.feature.spectral_centroid(y=X, sr=sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # 谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y=X))

    # 使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccsstd = np.std(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)
    mfccmax = np.max(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis=0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rms(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features

def extract_features(file: str, pad: bool = False) -> np.ndarray:
    X, sample_rate = librosa.load(file, sr=None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)

def get_max_min(files: list) -> Tuple[float]:
    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr=None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_

def get_data_path(data_path: str, class_labels: list) -> list:
    """
    获取所有音频的路径

    Args:
        data_path (str): 数据集文件夹路径
        class_labels (list): 情感标签
    Returns:
        wav_file_path (list): 所有音频的路径
    """
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)

    # 遍历文件夹
    for _, directory in enumerate(class_labels):
        os.chdir(directory)

        # 读取该文件夹下的音频
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path
