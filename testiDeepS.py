import sys
import os
import numpy
import pdb
import argparse

from BERT import BERTmodeling
from seq_motifs import *
import structure_motifs
from rnashape_structure import run_rnashape
import matplotlib

## TRAIN -> MOTIF

## python ideeps.py --train=True
# --data_file=datasets/clip/10_PARCLIP_ELAVL1A_hg19/30000/training_sample_0/sequences.fa.gz --model_dir=models

##python ideeps.py --predict=True
# --data_file=datasets/clip/10_PARCLIP_ELAVL1A_hg19/30000/test_sample_0/sequences.fa.gz
# --model_dir=models --out_file=YOUR_OUTFILE

#python ideeps.py --motif=True
# --data_file=datasets/clip/10_PARCLIP_ELAVL1A_hg19/30000/test_sample_0/sequences.fa.gz --model_dir=models --motif_dir=YOUR_MOTIF_DIR

from transformers import BertModel, BertConfig, BertTokenizer, TFBertModel
import tensorflow as tf


def test():
    #templayer = BERTmodeling.transformer_model(tf.zeros(shape=[10, 2, 111], dtype=tf.int32))

    print("DOES IT WORK: ", tf.config.list_physical_devices())

    #config = BertConfig.from_json_file('./3-new-12w-0/config.json')
    #token_config = BertConfig.from_json_file('./3-new-12w-0/tokenizer_config.json')
    #tokenizer = BertTokenizer('./3-new-12w-0/vocab.txt')
    #model = BertModel.from_pretrained("./3-new-12w-0/pytorch_model.bin", config=config, from_pt=True)

    #model = TFBertModel.from_pretrained("./3-new-12w-0/pytorch_model.bin", from_pt=True, config=config)
    print("DONE")



if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #args = parse_arguments(parser)
    test()


