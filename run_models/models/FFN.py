import random
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pickle
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import f1_score
import numpy as np
from models.LR import create_data_sent
from models.LSTM import create_data_sent_nest_onehot
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score
import warnings
warnings.filterwarnings("ignore")

#https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097

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

class FFNClassifier(nn.Module):

    def __init__(self, no_features, n_classes):
        """"""

        super(FFNClassifier, self).__init__()

        #dimensionalities
        self.input_dim = no_features
        self.n_classes = n_classes


        self.ffn = nn.Sequential(nn.Linear(self.input_dim, (self.input_dim//2)),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.3),
                                   nn.Linear((self.input_dim//2), (self.input_dim//2)),
                                   nn.ReLU(),
                                   nn.Dropout(p=0.3),
                                   nn.Linear((self.input_dim//2), self.n_classes),
                                   )


    def forward(self, bow_features):

        output = self.ffn(bow_features)

        return output

def get_metrics(true, pred):

    rec = recall_score(true, pred)
    f1_minority = f1_score(true, pred, labels=[1])
    f1_micro = f1_score(true, pred, average='micro')
    prec = precision_score(true, pred)

    return prec, rec, f1_minority, f1_micro


def get_results_ffn(features_dict, batch_size, epochs, learning_rate, weights, features, embeddings):

    seed_all(10)

    import warnings
    warnings.filterwarnings("ignore")


        ########################

    with open('labels_for_ffn.p', 'rb') as handle:
        labels2 = pickle.load(handle)

    if features != None:
        f,l = create_data_sent(features_dict, features)
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoder = enc.fit(f)
        encoder_trans = enc.transform(f)

    if embeddings == None:
        feature_dim = encoder_trans.shape[1]
        pcc_dataset_bow = BoWDataset(input_features=encoder_trans, labels_list=labels2)

    elif embeddings == 'sent':
        sent_embeddings = (torch.load('all_sents_nested.pt')).numpy()
        if features != None:
            sparse_plus_norm = np.concatenate((sent_embeddings, encoder_trans), axis=1)
            pcc_dataset_bow = BoWDataset(input_features=sparse_plus_norm, labels_list=labels2)
            feature_dim = sparse_plus_norm.shape[1]
            #print("dim", feature_dim)
        else:
            pcc_dataset_bow = BoWDataset(input_features=sent_embeddings, labels_list=labels2)
            feature_dim = sent_embeddings.shape[1]

    elif embeddings == 'doc':
        doc_embeddings = (torch.load('all_sents_nested_context.pt')).numpy()

        if features != None:
            sparse_plus_cont = np.concatenate((doc_embeddings, encoder_trans), axis=1)
            pcc_dataset_bow = BoWDataset(input_features=sparse_plus_cont, labels_list=labels2)
            feature_dim = sparse_plus_cont.shape[1]
        else:
            pcc_dataset_bow = BoWDataset(input_features=doc_embeddings, labels_list=labels2)
            feature_dim = doc_embeddings.shape[1]


    split_train_, split_test_ = \
                    random_split(pcc_dataset_bow, [1325,569])
    train_dataloader = DataLoader(split_train_, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)
    test_dataloader = DataLoader(split_test_, batch_size=batch_size, worker_init_fn=seed_worker, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_class = 2
    model = FFNClassifier(feature_dim, num_class).to(device)

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0

        for idx, sample_batch in enumerate(dataloader):
            optimizer.zero_grad()
            #print(sample_batch['labels'].shape)
            predicted_label = model(sample_batch['features'].float())
            #print(predicted_label.shape, sample_batch['labels'].shape)
            loss = criterion(predicted_label, sample_batch['labels'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            predicted_label_ = torch.sigmoid(predicted_label)
            #print(predicted_label_.shape, predicted_label_)

            total_acc += (predicted_label_.argmax(1) == sample_batch['labels'].argmax(1)).sum().item()
            total_count += sample_batch['labels'].size(0)
            total_acc, total_count = 0, 0

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        f1_scores, precs, recs, f1_micros = [], [], [], []

        with torch.no_grad():
            for idx, sample_batch in enumerate(dataloader):
                predicted_label = model(sample_batch['features'].float())
                predicted_label_ = torch.sigmoid(predicted_label)

                total_acc += (predicted_label_.argmax(1) == sample_batch['labels'].argmax(1)).sum().item()
                total_count += sample_batch['labels'].size(0) #* sample_batch['labels'].size(0)
                p,r,f1_min, f1_mic = get_metrics(torch.flatten(sample_batch['labels'].argmax(1)), torch.flatten(predicted_label_.argmax(1)))
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

    torch.save(model.state_dict(), 'best_ffn.pt')

    return np.mean(f1_mics), np.mean(ps), np.mean(rs), np.mean(f1s), accu_val_5, ps_5, rs_5, f1s_5
