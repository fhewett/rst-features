from sklearn.metrics import precision_score, recall_score, make_scorer, f1_score
import numpy as np
import torch
from collections import Counter
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
#from sklearn.metrics import f1_score, classification_report
#from sklearn.tree import DecisionTreeClassifier

from more_itertools import powerset

prec_scorer = make_scorer(precision_score)
rec_scorer = make_scorer(recall_score)
f1_scorer = make_scorer(f1_score, average='micro')
f1_minority = make_scorer(f1_score, pos_label=1, average='binary')

RANDOM_SEED = 0

clf = Pipeline([
    ('feature_encoder', OneHotEncoder(handle_unknown = 'ignore')),

  ('log_reg', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED))
])

embed_clf_scale = Pipeline([
    ('scaler', StandardScaler()),

  ('log_reg', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, penalty='l1', solver='liblinear'))
])

embed_clf = Pipeline([

  ('log_reg', LogisticRegression(class_weight='balanced', random_state=RANDOM_SEED, penalty='l1', solver='liblinear'))
])

#features, labels = [], []
def create_data_sent(features_dict, chosen_features, binary=True):

    subfeatures, sublabels = [], []
    for text in features_dict:
        if text == 'maz-12666':
            continue
        else:
        #subfeatures, sublabels = [], []
            for item in features_dict[text]:

                #for edu in range(len(item['edu_ids'])):
                feats = []
                for feat in chosen_features:
                    feats.append(item[feat][0])
                    #subfeatures.append((item['s_or_n'][edu], item['relations'][edu], item['depth_scores'][edu], item['most_nuclear'][edu]))
                subfeatures.append(tuple(feats))
                if binary:
                    if item['importance'][0] != 0:
                        sublabels.append(1)
                    else:
                        sublabels.append(0)
                else:
                    sublabels.append(item['importance'][0])


    if len(chosen_features) == 1:
        feats = np.array(subfeatures).reshape(-1,1)

    else:
        feats = np.array(subfeatures)

    return feats, sublabels

#INPUT: FEATURES

def LR_model(feature_dict, input_features, embeddings): #feature_dict is all_texts_sent

    if embeddings == None:
        f,l = create_data_sent(feature_dict, input_features)
        clfs = clone(clf)
        scores = cross_validate(clfs, f,l, cv=10,
                        scoring=({'f1_minority':f1_minority, 'precision': prec_scorer,
                                                         'recall': rec_scorer, 'f1': f1_scorer}))

    if embeddings == 'sent':
        sent_embeddings = (torch.load('/Users/freya.hewett/neural-seg-class/all_sents_nested.pt')).numpy()
        clfs = clone(embed_clf)
        if input_features != ['None']:
            f,l = create_data_sent(feature_dict, input_features)
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            f_enc = enc.fit_transform(f)
            comb = np.concatenate((sent_embeddings,f_enc), axis=1)

            scores = cross_validate(clfs, comb,l, cv=10,
                        scoring=({'f1_minority':f1_minority, 'precision': prec_scorer,
                                                         'recall': rec_scorer, 'f1': f1_scorer}))
        else:
            with open('/Users/freya.hewett/neural-seg-class/flat_labels.p', 'rb') as handle:
                l = pickle.load(handle)
            scores = cross_validate(clfs, sent_embeddings,l, cv=10,
                                    scoring=({'f1_minority':f1_minority, 'precision': prec_scorer,
                                                                     'recall': rec_scorer, 'f1': f1_scorer}))

    if embeddings == 'doc':
        doc_embeddings = (torch.load('/Users/freya.hewett/neural-seg-class/all_sents_nested_context.pt')).numpy()
        clfs = clone(embed_clf)
        if input_features != ['None']:
            f,l = create_data_sent(feature_dict, input_features)
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            f_enc = enc.fit_transform(f)
            comb = np.concatenate((doc_embeddings,f_enc), axis=1)

            scores = cross_validate(clfs, comb,l, cv=10,
                        scoring=({'f1_minority':f1_minority, 'precision': prec_scorer,
                                                         'recall': rec_scorer, 'f1': f1_scorer}))
        else:
            with open('/Users/freya.hewett/neural-seg-class/flat_labels.p', 'rb') as handle:
                l = pickle.load(handle)
            scores = cross_validate(clfs, doc_embeddings,l, cv=10,
                                    scoring=({'f1_minority':f1_minority, 'precision': prec_scorer,
                                                                     'recall': rec_scorer, 'f1': f1_scorer}))


    return np.mean(scores['test_f1']), np.mean(scores['test_precision']), np.mean(scores['test_recall']), np.mean(scores['test_f1_minority'])
