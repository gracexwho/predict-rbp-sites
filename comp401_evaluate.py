import argparse
import gzip
import pdb
import random

from numpy import loadtxt
from LSTM import read_seq
from LSTM import read_structure
from LSTM import load_label_seq
from LSTM import read_seq_new


# TODO: Compare std between probabilities of LSTM vs transformers prediction.txt on testing dataset
# TODO: Run test_ideeps on the training dataset and see prediction accuracy
#


def read_prediction_txt(file):
    lines = loadtxt(file, delimiter="\n", unpack=False)
    return lines




tf = read_prediction_txt("models/transformers/prediction_transformers.txt")
lstm = read_prediction_txt("models/LSTM/prediction_LSTM.txt")

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


data = load_data_file("2_PARCLIP_AGO2MNASE_hg19/30000/test_sample_0/sequences.fa.gz")
#np.savetxt("data_y.txt", data["Y"], delimiter="\n")
classes = data["Y"]



