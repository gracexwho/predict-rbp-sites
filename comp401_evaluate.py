import argparse
import gzip
import pdb
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

from numpy import loadtxt
from LSTM import read_seq
from LSTM import read_structure
from LSTM import load_label_seq
from LSTM import read_seq_new



def read_txt(file):
    lines = loadtxt(file, delimiter="\n", unpack=False)
    return lines


def load_data_file(inputfile, seq = True, onlytest = False):
    """
        Load data matrices from the specified folder.
    """
    path = os.path.dirname(inputfile)
    if len(path):
        path = './'
    data = dict()
    if seq:
        tmp = []
        tmp.append(read_seq(inputfile))
        seq_onehot, structure = read_structure(inputfile, path)
        tmp.append(seq_onehot)
        data["seq"] = tmp
        data["structure"] = structure
    if onlytest:
        data["Y"] = []
    else:
        data["Y"] = load_label_seq(inputfile)

    return data


#data = load_data_file("2_PARCLIP_AGO2MNASE_hg19/30000/test_sample_0/sequences.fa.gz")
#np.savetxt("data_y.txt", data["Y"], delimiter="\n")
#labels = data["Y"]


tf = read_txt("models/transformers/prediction_transformers.txt")
lstm = read_txt("models/LSTM/prediction_LSTM.txt")
labels = read_txt("data_y.txt")

tf_1 = 0
tf_0 = 0
lstm_1 = 0
lstm_0 = 0
count_0 = 0
count_1 = 0

for index, l in enumerate(labels):
    if l == 0:
        tf_0 += tf[index]
        lstm_0 += lstm[index]
        count_0 += 1
    else:
        tf_1 += tf[index]
        lstm_1 += lstm[index]
        count_1 += 1

tf_1 = tf_1/count_1
tf_0 = tf_0/count_0
lstm_1 = lstm_1/count_1
lstm_0 = lstm_0/count_0

print("TF")
print(tf_1)
print(tf_0)
print("MEAN", np.mean(tf))
print("STD", np.std(tf))
print("LSTM")
print(lstm_1)
print(lstm_0)
print("MEAN", np.mean(lstm))
print("STD", np.std(lstm))

#tf_fig = plt.boxplot(tf)
fig1, ax1 = plt.subplots()
#ax1.set_title('Box Plot of Transformer Probabilities')
ax1.boxplot(tf)
ax1.figure.savefig("tf_fig.png")
#tf_fig.savefig("tf_fig.png")

fig2, ax2 = plt.subplots()
#ax2.set_title('Box Plot of LSTM Probabilities')
ax2.boxplot(lstm)
ax2.figure.savefig("lstm_fig.png")
#lstm_fig.savefig("lstm_fig.png")


TP_tf = 0
count_TP_tf = 0
TP_lstm = 0
count_TP_lstm = 0

for index, t in enumerate(tf):
    if t >= np.mean(tf) + 2*np.std(tf):
        count_TP_tf += 1
        if (labels[index] == 1):
            TP_tf += 1



for index, t in enumerate(lstm):
    if t >= np.mean(lstm) + 2*np.std(lstm):
        count_TP_lstm += 1
        if (labels[index] == 1):
            TP_lstm += 1

TP_tf = TP_tf / count_TP_tf
TP_lstm = TP_lstm / count_TP_lstm

print("TRUE POSTIIVE TF: ", TP_tf)
print("TRUE POSITIVE LSTM: ", TP_lstm)

count_FN_tf = 0
count_FN_lstm = 0
FN_tf = 0
FN_lstm = 0

for index, t in enumerate(tf):
    if t <= np.mean(tf) - 2*np.std(tf):
        count_FN_tf += 1
        if (labels[index] == 1):
            FN_tf += 1

for index, t in enumerate(lstm):
    if t <= np.mean(lstm) - 2*np.std(lstm):
        count_FN_lstm += 1
        if (labels[index] == 1):
            FN_lstm += 1

FN_tf = FN_tf / count_FN_tf
FN_lstm = FN_lstm / count_FN_lstm

print("FALSE NEGATIVE TF: ", FN_tf)
print("FALSE NEGATIVE LSTM: ", FN_lstm)

print("SENSITIVITY TF: ", TP_tf / (TP_tf + FN_tf))
print("SENSITIVITY LSTM: ", TP_lstm / (TP_lstm + FN_lstm))


