import torch
import torch.nn as nn
from torch import optim
import time
import os
import argparse
import numpy as np
import pickle
from models.LR import LR_model
from models.LSTM import get_results_lstm
from models.FFN import get_results_ffn
from more_itertools import powerset

import matplotlib.pyplot as plt
import torch.nn as nn

if __name__ == "__main__":

    #print(torch.cuda.current_device())
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(123)
    #np.random.seed(0)
    parser = argparse.ArgumentParser()

    parser.add_argument('-json_features', type=str, default="../sent_level_pcc.json", help='JSON file with the features')

    parser.add_argument('-model', type=str, default="FFN", help='LR, FFN, LSTM')
    parser.add_argument('-lr', type=float, default=0.5, help='initial learning rate [default: 0.5]')

    parser.add_argument('-epochs', type=int, default=15, help='number of epochs for train [default: 15]')
    parser.add_argument('-batch_size', type=int, default= 64, help='batch size for training [default: 64]')

    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    # model
    parser.add_argument('-dropout', type=float, default=0.1, help='the probability for dropout [default: 0.1]')
    parser.add_argument('-weights', nargs='+', default=[0.3, 2.5], type=float, help='weights')
    parser.add_argument('-features', nargs='+', default=['s_or_n', 'relations', 'most_nuclear', 'sentence_length', 'position'], help='features (SN, rels, depth, most nuclear)')
    parser.add_argument('-embedding', type=str, default='sent', help='type of embedding (none, sent, doc)')
    parser.add_argument('-log_fn', type=str, default='log_file.txt', help='the name of the log file (default log_file.txt)')
    parser.add_argument('-log', action='store_true', default=False, help='log metrics/params in a file (default=False)')
    parser.add_argument('-all', action='store_true', default=False, help='go through all models')

    args = parser.parse_args()

    #if torch.cuda.current_device() != -1:
        #args.use_gpu = True

    #n_gpu = torch.cuda.device_count()
    all_texts_sent = json.load(open(args.json_features, 'r'))

    def print_results(f1_mic, precision, recall, f1_min, model_name_for_log, all_mode=False, epochs=args.epochs):
        if all_mode:
            if args.log:
                with open(args.log_fn, "a") as log_file:
                    if model_name_for_log == 'LR':
                        print("Model ",model_name_for_log, " Features ",
                                combi, " Embeddings ", embeds, file=log_file)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min, file=log_file)
                        print("Model ", model_name_for_log, " Features ",
                                combi, " Embeddings ", embeds)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min)

                    else:
                        print("Model ", model_name_for_log, " Batch size ", batch_size, " Epochs ", epochs, " LR ", lr, " Weights ", args.weights, " Features ",
                                combi, " Embeddings ", embeds, file=log_file)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min, file=log_file)

                        print("Model ", model_name_for_log, " Batch size ", batch_size, " Epochs ", epochs, " LR ", lr, " Weights ", args.weights, " Features ",
                                combi, " Embeddings ", embeds)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min)


        else:
            if args.log:
                with open("log_file.txt", "a") as log_file:
                    if model_name_for_log == 'LR':
                        print("Model ",model_name_for_log, " Features ",
                                                        combi, " Embeddings ", embeds, file=log_file)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min, file=log_file)
                        print("Model ", model_name_for_log, " Features ",
                                                        combi, " Embeddings ", embeds)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min)
                    else:
                        print("Model ", args.model, " Batch size ", args.batch_size, " Epochs ", epochs, " LR ", args.lr, " Weights ", args.weights, " Features ",
                        args.features, " Embeddings ", args.embedding, file=log_file)
                        print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min, file=log_file)
            else:
                if model_name_for_log == 'LR':

                    print("Model ", model_name_for_log, " Features ",
                                                                        combi, " Embeddings ", embeds)
                    print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min)

                else:

                    print("Model ", args.model, " Batch size ", args.batch_size, " Epochs ", epochs, " LR ", args.lr, " Weights ", args.weights, " Features ",
                        args.features, " Embeddings ", args.embedding)
                    print("F1 ", f1_mic, " Precision ", precision, " Recall ", recall, " F1 minority ", f1_min)

    if args.all:
        for combi in list(powerset(['s_or_n', 'relations', 'depth_scores', 'most_nuclear', 'sentence_length',
                           'position'])):
            for embeds in [None, 'sent', 'doc']:
                if combi == ():
                    combi = None
                if combi == None and embeds == None:
                    continue

                f1_mic, precision, recall, f1_min = LR_model(all_texts_sent, combi, embeds)
                model_name_for_log = 'LR'
                print_results(f1_mic, precision, recall, f1_min, model_name_for_log, all_mode=True)

                for batch_size in [8,12,16,32,64]:
                    for lr in [0.5, 0.8, 1]:
                        epochs = 15
                        f1_mic, precision, recall, f1_min, f1_mic_5, precision_5, recall_5, f1_min_5 = get_results_lstm(all_texts_sent, batch_size, epochs, lr, args.weights, combi, embeds)
                        model_name_for_log = 'LSTM'
                        print_results(f1_mic, precision, recall, f1_min, model_name_for_log, all_mode=True, epochs=epochs)
                        print_results(f1_mic_5, precision_5, recall_5, f1_min_5, model_name_for_log, all_mode=True, epochs=5)

                        f1_mic, precision, recall, f1_min, f1_mic_5, precision_5, recall_5, f1_min_5  = get_results_ffn(all_texts_sent, batch_size, epochs, lr, args.weights, combi, embeds)
                        model_name_for_log = 'FFN'
                        print_results(f1_mic, precision, recall, f1_min, model_name_for_log, all_mode=True, epochs=epochs)
                        print_results(f1_mic_5, precision_5, recall_5, f1_min_5, model_name_for_log, all_mode=True, epochs=5)


    if args.model == "LR":
        f1_mic, precision, recall, f1_min = LR_model(all_texts_sent, args.features, args.embedding)
        print_results(f1_mic, precision, recall, f1_min, "LR", all_mode=False)

    if args.model == "LSTM":
        f1_mic, precision, recall, f1_min, _, _, _, _ = get_results_lstm(all_texts_sent, args.batch_size, args.epochs, args.lr, args.weights, args.features, args.embedding)
        print_results(f1_mic, precision, recall, f1_min, "LSTM", all_mode=False)

    if args.model == "FFN":
        f1_mic, precision, recall, f1_min, _, _, _, _ = get_results_ffn(all_texts_sent, args.batch_size, args.epochs, args.lr, args.weights, args.features, args.embedding)
        print_results(f1_mic, precision, recall, f1_min, "FFN", all_mode=False)
