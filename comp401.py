import argparse
import gzip
import pdb
import random

import subprocess as sp

from rnashape_structure_without_memory_issue import run_rnashape

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
import os

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import np_utils


### TODO: Delete structure.gz and run on test_sample_0 rnashapes shit all over again



### TRAIN
#### python comp401.py --train=True --data_file=2_PARCLIP_AGO2MNASE_hg19/30000/training_sample_0/sequences.fa.gz --model_dir=models

### TEST
## before this: delete structure.gz because they need to do it again
## python comp401.py --predict=True --data_file=2_PARCLIP_AGO2MNASE_hg19/30000/test_sample_0/sequences.fa.gz --model_dir=models --out_file=prediction.txt

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        #self.ffn = tf.keras.Sequential(
        #    [tf.layers.Dense(ff_dim, activation="relu"), tf.layers.Dense(embed_dim), ]
        #)
        self.ffn1 = layers.Dense(ff_dim, activation="relu")
        self.ffn2 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


### TODO: Major

def final_transformers(data_file, model_dir, batch_size=50, nb_epoch=30):
    training_data = load_data_file(data_file)

    seq_hid = 16
    struct_hid = 16
    # pdb.set_trace()
    train_Y = training_data["Y"]
    print(len(train_Y))
    # pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    # pdb.set_trace()

    cnn_train = []
    cnn_validation = []
    seq_data = training_data["seq"][0]
    # pdb.set_trace()
    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice]
    struct_data = training_data["seq"][1]
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)

    total_hid = seq_hid + struct_hid



    print("SHAPE OF INPUTS")

    #print(len(cnn_train))
    #print(len(cnn_train[0]))


    #test_seq = np.array(seq_train)
    #test_struct = np.array(struct_train)

    #print("SEQ TRAIN")
    #print(np.shape(test_seq))

    #print("STRUCT TRAIN")
    #print(np.shape(test_struct))

    ### set_cnn_network

    #
    ##SEQ TRAIN
    #(24000, 111, 4)
    ##STRUCT TRAIN
    #(24000, 111, 6)

    nbfilter = 16

    model_inputs_seq = tf.keras.Input(shape=(111, 4), batch_size=batch_size)
    convlayer_seq = tf.keras.layers.Conv1D(filters=nbfilter, kernel_size=10, padding="valid", activation='relu')  # input_dim=input_dim
    x_seq = convlayer_seq(model_inputs_seq)
    print("X_seq")
    print(x_seq.shape)
    x_seq = tf.keras.layers.MaxPool1D(pool_size=3)(x_seq)
    x_seq = tf.keras.layers.Dropout(0.5)(x_seq)


    model_inputs_struct = tf.keras.Input(shape=(111, 6), batch_size=batch_size)
    convlayer_struct = tf.keras.layers.Conv1D(filters=nbfilter, kernel_size=10, padding="valid", activation='relu')  # input_dim=input_dim
    x_struct = convlayer_struct(model_inputs_struct)
    print("X_struct")
    print(x_struct.shape)
    x_struct = tf.keras.layers.MaxPool1D(pool_size=3)(x_struct)
    x_struct = tf.keras.layers.Dropout(0.5)(x_struct)

    x = layers.concatenate([x_seq, x_struct])

    ### get_transformers_network_2

    transformer_block = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32)
    x = transformer_block(x)
    x = tf.keras.layers.Flatten()(x)   ##TODO
    x = tf.keras.layers.Dropout(0.10)(x)
    x = tf.keras.layers.Dense(nbfilter * 2, activation='relu')(x)



    #rasTensor' object has no attribute 'lower'



    ### run_network_new

    model_outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    ### TODO: possibly missing a layer to get from 50, 34, 2 -> 50, 2

    #### train_ideeps ####
    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder=encoder)


    model = tf.keras.Model(inputs=[model_inputs_seq, model_inputs_struct], outputs=model_outputs)

    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.summary()

    #     model = run_network_new(seq_net, total_hid, cnn_train, y, validation=cnn_validation, val_y=val_y,
    #                             batch_size=batch_size, nb_epoch=nb_epoch)
    # def run_network_new(model, total_hid, training, y, validation, val_y, batch_size=50, nb_epoch=30):

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    print('WHY IS THIS WRONG')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(np.shape(np.array(seq_train)))
    print(np.shape(np.array(struct_train)))
    print(np.shape(np.array(y)))

    model.fit(x=[seq_train, struct_train], y=y, batch_size=batch_size, epochs=30, verbose=0, validation_data=(cnn_validation, val_y),
                  callbacks=[earlystopper])

    #x = layers.concatenate([title_features, body_features, tags_input])

    #######

    #seq_net = get_transformers_network_2()
    seq_data = []



    model.save(os.path.join(model_dir, 'model.pkl'))





def set_cnn_model(input_dim, input_length):
    nbfilter = 16


    #  [?,1,4,111], [1,10,111,16].
    #print("INPUT DIM", input_dim) 4 or 6
    #print("INPUT LENGTH", input_length) #111
    # input_dim
    # bad input shapes negative [?,1,4,111], [1,10,111,16].
    #(24000, 111, 4)
    #(24000, 111, 6)

    model_inputs = tf.keras.Input(shape=(input_length, input_dim))
    # 24000, 111 worked with warning
    # None, 24000, 111
    # 24000,
    # 24000
    # 24000, 111
    # input_length, input_dim
    # None, None, None
    # 24000, 111, input_dim
    # None, input_dim, inpu_length
    # (1,input_length, input_dim))
    # shape=(None,input_length)) -> WORKED
    # 24000,input_length -> WORKED
    convlayer = tf.keras.layers.Conv1D(filters=nbfilter, kernel_size=10, padding="valid", activation='relu', input_dim=input_dim) #input_dim=input_dim
    x = convlayer(model_inputs)
    print("X")
    print(x.shape)
    #x = tf.keras.Activation('relu')(x)
    x = tf.keras.layers.MaxPool1D(pool_size=3)(x)
    model_outputs = tf.keras.layers.Dropout(0.5)(x)
    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
    model.summary()

    ########
    #model = Sequential()
    #model.add(Convolution1D(filters=nbfilter, kernel_size=10, input_shape=(input_length, input_dim), padding="valid"))

    ##model.add(Convolution1D(input_dim=input_dim,input_length=input_length,
                            #nb_filter=nbfilter,
                            #filter_length=10,
                            #border_mode="valid",
                            #activation="relu",
                            #subsample_length=1))
    #model.add(Activation('relu'))
    #model.add(MaxPooling1D(pool_size=3))

    #model.add(Dropout(0.5))

    #print("CNN MODEL OUTPUT SHAPE: ")
    print(model.output_shape) #(None, 34, 16)

    return model



def get_transformers_network_2():
    nbfilter = 16
    print('configure cnn network')

    seq_model = set_cnn_model(4, 111).outputs[0]
    struct_model = set_cnn_model(6, 111).outputs[0]

    # print(tf.keras.backend.is_keras_tensor(seq_model))
    # print(seq_model.shape)
    # combined_model = tf.keras.Model(model_inputs, [seq_model, struct_model])

    # model_inputs = tf.keras.Input(shape=(None,))
    combined_model = tf.keras.layers.Concatenate(axis=-1)([seq_model, struct_model])

    model_inputs = tf.keras.Input(shape=combined_model.shape)

    # x = combined_model(model_inputs)
    transformer_block = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32)
    x = transformer_block(model_inputs)
    x = tf.keras.layers.Dropout(0.10)(x)
    model_outputs = tf.keras.layers.Dense(nbfilter * 2, activation='relu')(x)
    model = tf.keras.Model(model_inputs, model_outputs)

    model.summary()

    # ~~~~~~~~~~~~~~~~~~~`
    ## pdb.set_trace()
    # model = Sequential()
    # model.add(Concatenate([seq_model, struct_model]))

    ## model.add(Merge([seq_model, struct_model], mode='concat', concat_axis=1))

    ####model.add(Bidirectional(LSTM(2 * nbfilter)))

    # batch_size = 50
    # (50, ,  , 10)
    # print("MODEL OUTPUT SHAPE JUST BEFORE TRANSFORMERS: ")
    # print(model.output_shape)

    # transformer_block = TransformerBlock(32, 2, 32)
    # model.add(transformer_block)
    # model.add(Dropout(0.10))
    # model.add(Dense(nbfilter * 2, activation='relu'))
    # print(model.output_shape)

    return model




def run_network_new(model, total_hid, training, y, validation, val_y, batch_size=50, nb_epoch=30):

    #print("FINAL MODEL SUMMARY")
    #model.summary()

    #MODEL OUTPUT SHAPE
    #(None, None, 34, 32)
    #(None, None, 34, 32)
    #total hid  32
    print("TRAINING SHAPE")
    print(len(training))
    print(training[0].shape, flush=True)
    print(training[1].shape, flush=True)

    seq_model = set_cnn_model(4, 111)
    struct_model = set_cnn_model(6, 111)
    seq_net = model

    model_inputs1 = tf.keras.Input(111, 4) #(None, None, 32)
    model_inputs2 = tf.keras.Input(111, 6)

    x1 = seq_model(model_inputs1)
    x2 = struct_model(model_inputs2)

    concatted = tf.keras.layers.concatenate([x1, x2])

    x = seq_net(concatted)

    model_outputs = tf.keras.layers.Dense(2, activation='softmax')(x)
    new_model = tf.keras.models.Model(inputs=[model_inputs1, model_inputs2], outputs=model_outputs, name="Complete Model")
    new_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    new_model.summary()

    earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)
    new_model.fit(training, y, batch_size=batch_size, epochs=nb_epoch, verbose=0, validation_data=(validation, val_y),
                  callbacks=[earlystopper])
    # ~~~~~~~~~~~~~~~
    # model.add(Dense(2, input_shape=(total_hid,)))
    # model.add(Activation('softmax'))

    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # pdb.set_trace()
    print('model training')

    # earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=0)

    # model.fit(training, y, batch_size=batch_size, epochs=nb_epoch, verbose=0, validation_data=(validation, val_y), callbacks=[earlystopper])

    return model


def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder


def get_RNA_seq_concolutional_array(seq, motif_len=10):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    # for seq in seqs:
    # for key, seq in seqs.iteritems():
    half_len = motif_len // 2
    row = (len(seq) + half_len * 2)
    new_array = np.zeros((row, 4))
    for i in range(half_len):
        new_array[i] = np.array([0.25] * 4)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.25] * 4)

    # pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len - 1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25] * 4)
            continue
        # if val == 'N' or i < motif_len or i > len(seq) - motif_len:
        #    new_array[i] = np.array([0.25]*4)
        # else:
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        # data[key] = new_array
    return new_array


def get_RNA_structure_concolutional_array(seq, fw, structure=None, motif_len=10):
    # print(fw)
    ####
    if fw is None:
        # also implies structure != None
        struc_en = structure
    else:
        seq = seq.replace('T', 'U')
        struc_en = run_rnashape(seq)
        struct_en = struc_en + '\n'
        fw.write(struct_en)  #### bytes-like object instead of str, so encode
    ####

    alpha = 'FTIHMS'
    half_len = motif_len // 2
    row = (len(struc_en) + half_len * 2)
    new_array = np.zeros((row, 6))
    for i in range(half_len):
        new_array[i] = np.array([0.16] * 6)

    for i in range(row - half_len, row):
        new_array[i] = np.array([0.16] * 6)

    for i, val in enumerate(struc_en):
        i = i + motif_len - 1
        if val not in alpha:
            new_array[i] = np.array([0.16] * 6)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()

    return new_array, struc_en


def read_rnashape(structure_file):
    struct_dict = {}
    with gzip.open(structure_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[:-1]
            else:
                strucure = line[:-1]
                struct_dict[name] = strucure

    return struct_dict

def read_seq(seq_file):
    seq_list = []
    seq = ''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                if len(seq):
                    seq_array = get_RNA_seq_concolutional_array(seq)
                    seq_list.append(seq_array)
                seq = ''
            else:
                seq = seq + str(line[:-1])
        if len(seq):
            seq_array = get_RNA_seq_concolutional_array(seq)
            seq_list.append(seq_array)

    return np.array(seq_list)




def read_structure(seq_file, path):
    seq_list = []
    structure_list = []
    struct_exist = False
    if not os.path.exists(path + '/structure.gz'):
        print("STRUCTURE DOES NOT EXIST")
        fw = gzip.open(path + '/structure.gz', 'wt')
    else:
        print("STRUCTURE EXISTS")
        fw = None
        struct_exist = True
        struct_dict = read_rnashape(path + '/structure.gz')
        #pdb.set_trace()


    #### TESTING ###

    #def filter_chr1(test):
    #    if (test.find('chr1,+,') != -1):
    #        return True
    #    else:
    #        return False

    #filtered = filter(filter_chr1, struct_dict.keys())

    #for f in filtered:
    #    print(f)

    seq = ''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line
                if len(seq):
                    if struct_exist:
                        structure = struct_dict[old_name[:-1]] ######string
                        seq_array, struct = get_RNA_structure_concolutional_array(seq, fw, structure = structure)
                    else:
                        fw.write(old_name)
                        seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
                    seq_list.append(seq_array)
                    structure_list.append(struct)
                old_name = name
                seq = ''
            else:
                seq = seq + line[:-1]
        if len(seq):
            if struct_exist:
                structure = struct_dict[old_name[:-1]]
                seq_array, struct = get_RNA_structure_concolutional_array(seq, fw, structure = structure)
            else:
                fw.write(old_name)
                seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
            #seq_array, struct = get_RNA_structure_concolutional_array(seq, fw)
            seq_list.append(seq_array)
            structure_list.append(struct)
    if fw:
        fw.close()
    return np.array(seq_list), structure_list

def load_label_seq(seq_file):
    label_list = []
    seq = ''
    with gzip.open(seq_file, 'rt') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
                posi_label = name.split(';')[-1]
                label = posi_label.split(':')[-1]
                label_list.append(int(label))
    return np.array(label_list)

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

    #cmd = "docker stop rnashapes"
    #out = sp.check_output(cmd, shell=True)

    return data


def split_training_validation(classes, validation_size=0.2, shuffle=False):
    """split sampels based on balnace classes"""
    num_samples = len(classes)
    classes = np.array(classes)
    classes_unique = np.unique(classes)
    num_classes = len(classes_unique)
    indices = np.arange(num_samples)
    # indices_folds=np.zeros([num_samples],dtype=int)
    training_indice = []
    training_label = []
    validation_indice = []
    validation_label = []
    for cl in classes_unique:
        indices_cl = indices[classes == cl]
        num_samples_cl = len(indices_cl)

        # split this class into k parts
        if shuffle:
            random.shuffle(indices_cl)  # in-place shuffle

        # module and residual
        num_samples_each_split = int(num_samples_cl * validation_size)
        res = num_samples_cl - num_samples_each_split

        training_indice = training_indice + [val for val in indices_cl[num_samples_each_split:]]
        training_label = training_label + [cl] * res

        validation_indice = validation_indice + [val for val in indices_cl[:num_samples_each_split]]
        validation_label = validation_label + [cl] * num_samples_each_split

    training_index = np.arange(len(training_label))

    random.shuffle(training_index)
    training_indice = np.array(training_indice)[training_index]
    training_label = np.array(training_label)[training_index]

    validation_index = np.arange(len(validation_label))
    random.shuffle(validation_index)
    validation_indice = np.array(validation_indice)[validation_index]
    validation_label = np.array(validation_label)[validation_index]

    return training_indice, training_label, validation_indice, validation_label


def train_ideeps(data_file, model_dir, batch_size=50, nb_epoch=30):
    training_data = load_data_file(data_file)

    seq_hid = 16
    struct_hid = 16
    # pdb.set_trace()
    train_Y = training_data["Y"]
    print(len(train_Y))
    # pdb.set_trace()
    training_indice, training_label, validation_indice, validation_label = split_training_validation(train_Y)
    # pdb.set_trace()

    cnn_train = []
    cnn_validation = []
    seq_data = training_data["seq"][0]
    # pdb.set_trace()
    seq_train = seq_data[training_indice]
    seq_validation = seq_data[validation_indice]
    struct_data = training_data["seq"][1]
    struct_train = struct_data[training_indice]
    struct_validation = struct_data[validation_indice]
    cnn_train.append(seq_train)
    cnn_train.append(struct_train)
    cnn_validation.append(seq_validation)
    cnn_validation.append(struct_validation)


    print("SHAPE OF INPUTS")
    print("TRAINING SHAPE")
    print(cnn_train.shape)
    print(cnn_train[0].shape)

    seq_net = get_transformers_network_2()
    seq_data = []

    y, encoder = preprocess_labels(training_label)
    val_y, encoder = preprocess_labels(validation_label, encoder=encoder)

    total_hid = seq_hid + struct_hid
    model = run_network_new(seq_net, total_hid, cnn_train, y, validation=cnn_validation, val_y=val_y,
                            batch_size=batch_size, nb_epoch=nb_epoch)

    model.save(os.path.join(model_dir, 'model.pkl'))


def test_ideeps(data_file, model_dir, outfile='prediction.txt', onlytest=True):
    test_data = load_data_file(data_file, onlytest=onlytest)
    #print(len(test_data))
    if not onlytest:
        true_y = test_data["Y"].copy()

    print('predicting')

    testing = test_data["seq"]  # it includes one-hot encoding sequence and structure
    structure = test_data["structure"]

    model = load_model(os.path.join(model_dir, 'model.pkl'))

    #model.summary()

    print("TESTING SHAPE")
    #print(len(testing[1]))
    print(test_data.keys())

    print(len(testing))
    print(len(testing[0]))
    print(len(testing[1]))

    print(len(structure))
    print(len(structure[0]))

    #d1 = tf.data.Dataset.from_tensor_slices(testing)
    #d2 = tf.data.Dataset.from_tensor_slices(structure)
    #zipped_input = tf.data.Dataset.zip((d1, d2)).batch(50)
    pred = model.predict(x=testing, batch_size=50)

    fw = open(outfile, 'wb')
    myprob = "\n".join(map(str, pred[:, 1]))
    # fw.write(mylabel + '\n')
    fw.write(myprob.encode())
    fw.close()






def run_ideeps(parser):
    data_file = parser.data_file
    out_file = parser.out_file
    train = parser.train
    model_dir = parser.model_dir
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    motif = parser.motif
    motif_dir = parser.motif_dir

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if predict:
        train = False

    if train:
        print('model training')
        #train_ideeps(data_file, model_dir, batch_size=batch_size, nb_epoch=n_epochs)
        final_transformers(data_file, model_dir, batch_size=batch_size, nb_epoch=n_epochs)
    else:
        print('model prediction')
        test_ideeps(data_file, model_dir, outfile=out_file, onlytest=True)

    ### TODO
    #if motif:
    #    identify_motif(data_file, model_dir, motif_dir, onlytest=True)



def parse_arguments(parser):
    parser.add_argument('--data_file', type=str, metavar='<data_file>', required=True,
                        help='the sequence file used for training, it contains sequences and label (0, 1) in each head of sequence.')
    parser.add_argument('--train', type=bool, default=True, help='use this option for training model')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='The directory to save the trained models for future prediction')
    parser.add_argument('--predict', type=bool, default=False,
                        help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--out_file', type=str, default='prediction.txt',
                        help='The output file used to store the prediction probability of testing data')
    parser.add_argument('--motif', type=bool, default=False, help='Identify motifs using CNNs.')
    parser.add_argument('--motif_dir', type=str, default='motifs', help='The directory to save the identified motifs.')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='The size of a single mini-batch (default value: 50)')
    parser.add_argument('--n_epochs', type=int, default=30, help='The number of training epochs (default value: 30)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)
    run_ideeps(args)