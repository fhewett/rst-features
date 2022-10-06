from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import random
import numpy as np
from LR import create_data_sent
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score
import warnings
warnings.filterwarnings("ignore")

#https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
#for reproducibility
def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

class DocEmbeddingsDatasetSequence(Dataset):
    """"""

    def __init__(self, saved_tensor, labels_list):
        """
        Args:

        """
        self.embeddings_tensor = torch.load(saved_tensor)
        self.labels_list = labels_list

    def __len__(self):
        return self.embeddings_tensor.shape[0]

    def __getitem__(self, idx):

        sample = {'features': self.embeddings_tensor[idx], 'labels':self.labels_list[idx]}


        return sample

class BoWDataset(Dataset):
    """"""

    def __init__(self, input_features, labels_list):
        """
        Args:

        """
        self.input_features = input_features
        self.labels_list = labels_list

    def __len__(self):
        return self.input_features.shape[0]

    def __getitem__(self, idx):

        sample = {'features': self.input_features[idx], 'labels':self.labels_list[idx]}

        return sample

class DocumentLevelClassifier(nn.Module):

    def __init__(self, input_dim, n_classes): #doc_embeddings are the pooled embeddings
                                                                #created in the previous steps
            #doc_embeddings
        """"""

        super(DocumentLevelClassifier, self).__init__()

        #dimensionalities
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_dim = 512


        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=1,
                            bidirectional=True,
                           batch_first=True)
        #self.drop = nn.Dropout(p=0.5)

        self.fc = nn.Linear(2*self.hidden_dim, 2)

    def forward(self, features_batch):

        output, _ = self.lstm(features_batch)
        text_fea = self.fc(output)

        return text_fea


def get_metrics(true, pred):

    rec = recall_score(true, pred)
    f1_minority = f1_score(true, pred, labels=[1])
    f1_micro = f1_score(true, pred, average='micro')
    prec = precision_score(true, pred)

    return prec, rec, f1_minority, f1_micro


def create_data_sent_nest(features_dict, chosen_features, fit_encoder, feature_dim, binary=True):

    sublabels = []
    subfeatures = np.zeros((len(features_dict),18,feature_dim))

    for i, text in enumerate(features_dict):
        subsubfeatures = []
        if text == 'maz-12666':
            continue
        else:
            for item in features_dict[text]:

                feats = []
                for feat in chosen_features:
                    if type(item[feat][0]) != str:
                        feats.append(str(item[feat][0]))
                    else:
                        feats.append(item[feat][0])

                if binary:
                    if item['importance'][0] != 0:
                        sublabels.append(1)
                    else:
                        sublabels.append(0)
                else:
                    sublabels.append(item['importance'][0])

                subsubfeatures.append(feats)

        # Pad then append
        encoded = fit_encoder.transform(subsubfeatures)
        padded = np.pad(encoded, pad_width=((0, (18-len(encoded))), (0,0)), mode='constant')
        expanded = np.expand_dims(padded, axis=0)
        subfeatures[i] = expanded

    if len(chosen_features) == 1:
        feats = np.array(subfeatures).reshape(-1,1)

    else:
        feats = np.array(subfeatures)

    return feats, sublabels, subfeatures


def create_data_sent_nest_onehot(features_dict, chosen_features, feature_dim, fit_encoder, binary=True, encode=False):

    sublabels = []
    subfeatures = np.zeros((len(features_dict),18,feature_dim))

    for i, text in enumerate(features_dict):
        subsubfeatures = []
        if text == 'maz-12666':
            continue
        else:
            for item in features_dict[text]:
                feats = []
                for feat in chosen_features:
                    if type(item[feat][0]) != str:
                        feats.append(str(item[feat][0]))
                    else:
                        feats.append(item[feat][0])

                if binary:
                    if item['importance'][0] != 0:
                        sublabels.append(1)
                    else:
                        sublabels.append(0)
                else:
                    sublabels.append(item['importance'][0])

                subsubfeatures.append(feats)

        # Pad then append
        if encode:
            encoded = fit_encoder.transform(subsubfeatures)
        else:
            encoded = subsubfeatures
        padded = np.pad(encoded, pad_width=((0, (18-len(encoded))), (0,0)), mode='constant')
        #print(padded.shape, padded)
        expanded = np.expand_dims(padded, axis=0)
        #print(expanded.shape, expanded)
        subfeatures[i] = expanded

    feats = np.array(subfeatures)

    return feats, sublabels, subfeatures

def get_results_lstm(features_dict, batch_size, epochs, learning_rate, weights, features, embeddings):

    seed_all(10)

    with open('labels_nested_binarized.p', 'rb') as handle:
        labels2 = pickle.load(handle)

    if features != None:
        f,l = create_data_sent(features_dict, features)
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder = enc.fit(f)
        encoder_trans = enc.transform(f)

    if embeddings == None:
        feature_dim = encoder_trans.shape[1]
        _, _, feats = create_data_sent_nest(features_dict, features, encoder, encoder_trans.shape[1], binary=True)
        #print(feats.shape)
        pcc_dataset_bow = BoWDataset(input_features=feats, labels_list=labels2)

    elif embeddings == 'sent':
        sent_embeddings = (torch.load('padded_sequence_normal_sent.pt')).numpy()
        if features != None:
            feats, labs, _ = create_data_sent_nest_onehot(features_dict, features, encoder_trans.shape[1], encoder, encode=True)
            sparse_plus_norm = np.concatenate((sent_embeddings, feats), axis=2)
            pcc_dataset_bow = BoWDataset(input_features=sparse_plus_norm, labels_list=labels2)
            feature_dim = sparse_plus_norm.shape[2]
        else:
            pcc_dataset_bow = BoWDataset(input_features=sent_embeddings, labels_list=labels2)
            feature_dim = sent_embeddings.shape[2]

    elif embeddings == 'doc':
        doc_embeddings = (torch.load('padded_sequence_context_sent.pt')).numpy()

        if features != None:
            feats, labs, _ = create_data_sent_nest_onehot(features_dict, features, encoder_trans.shape[1], encoder, encode=True)
            sparse_plus_cont = np.concatenate((doc_embeddings, feats), axis=2)
            pcc_dataset_bow = BoWDataset(input_features=sparse_plus_cont, labels_list=labels2)
            feature_dim = sparse_plus_cont.shape[2]
        else:
            pcc_dataset_bow = BoWDataset(input_features=doc_embeddings, labels_list=labels2)
            feature_dim = doc_embeddings.shape[2]


    split_train_, split_test_ = \
                    random_split(pcc_dataset_bow, [135, 32])
    train_dataloader = DataLoader(split_train_, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)
    test_dataloader = DataLoader(split_test_, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 2
    model = DocumentLevelClassifier(feature_dim, num_class).to(device)

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0

        for idx, sample_batch in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(sample_batch['features'].float())
            #print(predicted_label.shape, sample_batch['labels'].shape)
            loss = criterion(predicted_label, sample_batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            predicted_label_ = torch.sigmoid(predicted_label)

            total_acc += (predicted_label_.argmax(2) == sample_batch['labels'].argmax(2)).sum().item()
            total_count += sample_batch['labels'].size(1) * sample_batch['labels'].size(0)

            total_acc, total_count = 0, 0

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        f1_scores, precs, recs, f1_micros = [], [], [], []

        with torch.no_grad():
            for idx, sample_batch in enumerate(dataloader):
                predicted_label = model(sample_batch['features'].float())
                #loss = criterion(predicted_label, sample_batch['labels'])
                predicted_label_ = torch.sigmoid(predicted_label)

                total_acc += (predicted_label_.argmax(2) == sample_batch['labels'].argmax(2)).sum().item()
                total_count += sample_batch['labels'].size(1) * sample_batch['labels'].size(0)
                p,r,f1_min, f1_mic = get_metrics(torch.flatten(sample_batch['labels'].argmax(2)), torch.flatten(predicted_label_.argmax(2)))
                f1_scores.append(f1_min)
                precs.append(p)
                recs.append(r)
                f1_micros.append(f1_mic)

        return total_acc/total_count, f1_scores, precs, recs, f1_micros

    # Hyperparameters
    EPOCHS = epochs
    LR = learning_rate
    #BATCH_SIZE = 16 # batch size for training

    weights_ = torch.FloatTensor(weights)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights_)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None

    for epoch in range(1, EPOCHS + 1):
        #epoch_start_time = time.time()
        train(train_dataloader)
        accu_val, f1s, ps, rs, f1_mics = evaluate(test_dataloader)

        if epoch == 5:
            accu_val_5, ps_5, rs_5, f1s_5 = np.mean(f1_mics), np.mean(ps), np.mean(rs), np.mean(f1s)

        if total_accu is not None and total_accu > accu_val:
            scheduler.step()
        else:
            total_accu = accu_val

    return np.mean(f1_mics), np.mean(ps), np.mean(rs), np.mean(f1s), accu_val_5, ps_5, rs_5, f1s_5
