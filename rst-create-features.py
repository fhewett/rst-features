import re
import json
import os
import spacy
import argparse

nlp = spacy.load("de_core_news_sm")

from initialise import initialise_edu_dict, match_edus_to_sent
from depth import get_depth_scores
from annotations import extract_anno
from sentence_level import get_sent_level

def tokenize_text_no_punct(input_sent):
    """Tokenises using tokeniser from spacy, excludes punctuation"""

    doc = nlp(input_sent)
    tkzed = [token.text for token in doc if not token.is_punct]

    return tkzed

def create_dict_with_annos_depth(sent_to_edu_dict, edu_dict, anno_dict, depth_scores):
    """Converts feature information into one dictionary"""

    properties = list()
    for sent_no in sent_to_edu_dict:
        if len(sent_to_edu_dict[sent_no]) == 0:
            continue
        properties.append({'sentence': sent_no, 'edu_ids': [], 's_or_n': [], 'relations':[], 'most_nuclear': [],
                          'importance': [], 'depth_scores': [], 'edu_length': [], 'position': []})
        for edu_nos in sent_to_edu_dict[sent_no]:
            info = edu_dict[edu_nos]
            properties[sent_no]['s_or_n'].append(info[1])
            properties[sent_no]['relations'].append(info[2])
            properties[sent_no]['most_nuclear'].append(info[3])
            properties[sent_no]['importance'].append(anno_dict[sent_no])
            properties[sent_no]['depth_scores'].append(depth_scores[edu_nos])
            properties[sent_no]['edu_length'].append(len(tokenize_text_no_punct(info[0])))
            properties[sent_no]['position'].append(edu_nos/max(edu_dict))

        properties[sent_no]['no_of_edus'] = len(properties[sent_no]['s_or_n'])
        properties[sent_no]['edu_ids'] = sent_to_edu_dict[sent_no]

    return properties


def for_all_texts(path):

    all_texts = dict()
    dirs = os.listdir(path + 'rst/')
    for file in dirs:
        if file.endswith('.rs3'):
            try:
                ad1 = extract_anno(file, path)
            except KeyError:
                print("KeyError whilst processing annotations for ", file, " (implies that there are no annotations for this file)")
                continue

            edu_d = initialise_edu_dict(path  + 'rst/' + file)

            if "PotsdamCommentaryCorpus" in path:
                sent_2_edu, sents = match_edus_to_sent(file[:-4], path)
                #sents = get_gold_syntax_sentences(path + 'syntax/' + file[:-4] + '.xml')
            else:
                sent_2_edu, sents = match_edus_to_sent(file[:-4], path)

            depth_s = get_depth_scores(file, path)

            try:
                props = create_dict_with_annos_depth(sent_2_edu, edu_d, ad1, depth_s)
            except IndexError:
                print("IndexError whilst creating features for ", file)
                continue

            all_texts[file[:-4]] = props


    return all_texts

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus_path', type=str, default='PotsdamCommentaryCorpus/', help='path to where corpus is saved; default: PotsdamCommentaryCorpus/')
    parser.add_argument('-level', type=str, default='both', help='what level analysis (edu, sentence, or both; default: both)')
    args = parser.parse_args()

    #first create EDU level dict
    edu_level_dict = for_all_texts(args.corpus_path)

    if args.level == "edu":
        json.dump(edu_level_dict, open('edu_level_pcc.json', 'w'), ensure_ascii=False)
        exit()

    if args.level == "both":
        json.dump(edu_level_dict, open('edu_level_pcc.json', 'w'), ensure_ascii=False)

    #then create sentence level
    all_texts_d_sent = dict()

    for file in edu_level_dict:
        sent_level = list()
        try:
            #print(file, args.corpus_path, "now sent")
            updated = get_sent_level(file, args.corpus_path)
        except TypeError:
            print("TypeError whilst creating sentence level features for ", file)
            continue
        except IndexError:
            print("IndexError whilst creating sentence level features for ", file)
            continue
        for sent in edu_level_dict[file]:
            if sent['sentence'] in updated:
                sent['relations'] = [updated[sent['sentence']]['relations']]
                sent['s_or_n'] = [updated[sent['sentence']]['s_or_n']]
                sent['depth_scores'] = [max(sent['depth_scores'])]
                sent['most_nuclear'] = [max(sent['most_nuclear'])]
                sent['importance'] = [max(sent['importance'])]
                sent['position'] = [sent['sentence'] / len(edu_level_dict[file])]
                sent['sent_length'] = [sum(sent['edu_length'])]
                sent_level.append(sent)
            else:
                sent['depth_scores'] = [max(sent['depth_scores'])]
                sent['most_nuclear'] = [max(sent['most_nuclear'])]
                sent['importance'] = [max(sent['importance'])]
                sent['position'] = [sent['sentence'] / len(edu_level_dict[file])]
                sent['sent_length'] = [sum(sent['edu_length'])]
                sent_level.append(sent)
        all_texts_d_sent[file] = sent_level

    if args.level == "sentence" or args.level == "both":
        json.dump(all_texts_d_sent, open('sent_level_pcc.json', 'w'), ensure_ascii=False)
