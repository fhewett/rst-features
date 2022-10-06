import re
import html
from difflib import SequenceMatcher as SM
import xml.etree.ElementTree as ET

def initialise_nuc(filename):
    """Takes the rs3 filename as input
        init_dict : dictionary where keys are parent IDs, values are segment/group IDs
        all_seg_ids : a list of all EDU level segment IDs
        edu_dict: dictionary, keys are sentence IDs, values are sentence text"""

    init_dict, edu_dict = dict(), dict()
    all_seg_ids = list()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()

        group_id = r'<group id="([0-9]*)" type="(\S*?)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_group = re.compile(group_id)
        for (groupid, typetype, parent, relname) in re.findall(regex_group, read_file):
            if relname in ["conjunction", "contrast", "disjunction", "joint",
                               "list", "restatement-mn", "sequence", "span"]:
                if int(parent) not in init_dict:
                    init_dict[int(parent)] = []
                init_dict[int(parent)].append(int(groupid))


        regex5 = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_seg = re.compile(regex5)
        for (segid, segparent, segrelname) in re.findall(regex_seg, read_file):
            if segrelname in ["conjunction", "contrast", "disjunction", "joint",
                               "list", "restatement-mn", "sequence", "span"]:
                if int(segparent) not in init_dict:
                    init_dict[int(segparent)] = []
                init_dict[int(segparent)].append(int(segid))
                if int(segid) not in init_dict:
                    init_dict[int(segid)] = [int(segid)]
            all_seg_ids.append(int(segid)) #not including root segment!!

        text_regex = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)">[\s\n]*([\S ]*)<\/segment>'
        te_rx = re.compile(text_regex)
        for (seg_id, parent, rel, text) in re.findall(te_rx, read_file):
            edu_dict[int(seg_id)] = text


    return init_dict, all_seg_ids, edu_dict


def get_root_node_rst(filename):
    with open(filename, 'r') as myfile:
        data = myfile.read()

    group_no_parent = r'<group id="([0-9]*)" type="\S*?" \/>'
    gnp_regex = re.compile(group_no_parent)

    group_id = re.findall(gnp_regex, data)

    return int(group_id[0])

def find_path(graph, start, end, path=[]):

    """Helper function copied from GitHub
        graph : init_dict, dictionary with parents as keys, children (seg/group IDs) as values
        Takes a start and end value and finds the path through the graph"""

    path = path + [start]
    if start == end:
        return path
    if start not in graph:
        return None
    for node in graph[start]:
        if node not in path:
            newpath = find_path(graph, node, end, path)
            if newpath: return newpath

    return None

def collate_nuc_paths(all_seg_ids, init_dict, st):

    """ For each group parent (values in init_dict), finds the paths to any of the EDUs (i.e. all_seg_ids)"""

    es = list()

    for group_id in init_dict[st]:
        for e in all_seg_ids:
            one_path = find_path(init_dict, group_id, e)
            if one_path != None:
                es.append(e)

    return es


def collate_paths(all_seg_ids, init_dict, st):#init_dict): #st is a key from init_dict

    """ For each group parent (values in init_dict), finds the paths to any of the EDUs (i.e. all_seg_ids)
        Collates these paths"""

    output_dict = dict()

    rc = 0

    if st in all_seg_ids:
        total_paths = [st]
    else:
        total_paths = list()

    for group_id in init_dict[st]:
        for e in all_seg_ids:
            one_path = find_path(init_dict, group_id, e)
            if one_path != None:
                total_paths += one_path

    edu_paths = list()
    for pp in set(total_paths):
        if pp in all_seg_ids:
            rc += 1
            edu_paths.append(pp)

    return rc, edu_paths

def automate(id_parent_dict, all_seg_ids, init_dict, id_rel, span_segments):
    """"""

    path_dict = dict()
    for m in span_segments:
        path_dict[m] = []

    for rst_id in id_parent_dict:
        satellites, s_paths = collate_paths(all_seg_ids, init_dict, rst_id)
        nuclei, n_paths = collate_paths(all_seg_ids, init_dict, id_parent_dict[rst_id])
        for m in span_segments:
            if m in n_paths:
                if id_rel[rst_id] not in ["conjunction", "contrast", "disjunction", "joint",
                               "list", "restatement-mn", "sequence", "span"]:
                    path_dict[m].append((id_rel[rst_id], n_paths))

    return path_dict

def get_nuc_rels(path_dict):

    rel_dict = dict()

    for n in list(path_dict.keys()):

        length, rel = [], []
        for elem in path_dict[n]:
            rel.append(elem[0])
            length.append(len(elem[1]))

        rel_dict[n] = rel[length.index(min(length))]

    return rel_dict

def initialise(filename):
    """Takes the rs3 filename as input
        init_dict : dictionary where keys are parent IDs, values are segment/group IDs
        all_seg_ids : a list of all EDU level segment IDs
        nuc_sat_dict : dictionary where keys are segment/group IDs, values are relation name
        id_par : dictionary where keys are segment/group IDs, values are their parents,
                for multinucs key and value are the same"""


    init_dict, nuc_sat_dict, id_par = dict(), dict(), dict()
    all_seg_ids, span_segs = list(), list()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()

        group_id = r'<group id="([0-9]*)" type="(\S*?)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_group = re.compile(group_id)
        for (groupid, typetype, parent, relname) in re.findall(regex_group, read_file):
            if int(parent) not in init_dict:
                init_dict[int(parent)] = []
            init_dict[int(parent)].append(int(groupid))

            nuc_sat_dict[int(groupid)] = relname

            if relname not in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence"]:
                id_par[int(groupid)] = int(parent)
            else:
                #for multinuc
                id_par[int(groupid)] = int(groupid)

        regex5 = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_seg = re.compile(regex5)
        for (segid, segparent, segrelname) in re.findall(regex_seg, read_file):
            if segrelname == 'span':
                span_segs.append(int(segid))
            if int(segparent) not in init_dict:
                init_dict[int(segparent)] = []
            init_dict[int(segparent)].append(int(segid))
            if int(segid) not in init_dict:
                init_dict[int(segid)] = [int(segid)]
            all_seg_ids.append(int(segid)) #not including root segment!!
            nuc_sat_dict[int(segid)] = segrelname
            if segrelname not in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence"]:

                id_par[int(segid)] = int(segparent)

            else:
                #for multinuc
                id_par[int(segid)] = int(segid)

    return init_dict, all_seg_ids, nuc_sat_dict, id_par, span_segs


def init_and_automate(file_path):

    init_dict, seg_ids, nuc_sat, id_parent, span_segs = initialise(file_path)
    path_dict = automate(id_parent, seg_ids, init_dict, nuc_sat, span_segs)
    rel_d = get_nuc_rels(path_dict)

    return rel_d

def initialise_edu_dict(filename):
    """Takes the rs3 filename as input
        """

    cc_rst = get_root_node_rst(filename)
    init_dict_1, all_seg_ids_1, edu_dict_1 = initialise_nuc(filename)

    #before collating paths:
    if cc_rst in all_seg_ids_1:
        list_of_mn_segs = [cc_rst]
    else:
        list_of_mn_segs = collate_nuc_paths(all_seg_ids_1, init_dict_1, cc_rst)

    edu_dict = dict()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()
        text_regex = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)">[\s\n]*([\S ]*)<\/segment>'
        te_rx = re.compile(text_regex)
        rel_dict = init_and_automate(filename)
        for (seg_id, parent, rel, text) in re.findall(te_rx, read_file):
            if rel == 'span':
                n_or_s = 'N'
                relation = rel_dict[int(seg_id)]

            elif rel in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence"]:
                n_or_s = 'N'
                relation = rel
            else:
                n_or_s = 'S'
                relation = rel

            if int(seg_id) in list_of_mn_segs:
                mn = 1
            else:
                mn = 0

            text = html.unescape(text)
            edu_dict[int(seg_id)] = [text, n_or_s, relation, mn]

    return edu_dict

def get_edus(filename):

    edus, segids = list(), dict()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()

        regex5 = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)">[\s\n]*([\S ]*)<\/segment>'
        regex_seg = re.compile(regex5)
        for (segid, segparent, segrelname, segtext) in re.findall(regex_seg, read_file):
            edus.append(html.unescape(segtext))
            segids[html.unescape(segtext)] = int(segid)

    return edus, segids

def get_gold_syntax_sentences(xml_file):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    all_sentences = []
    for s in root[0]:
        for child in s[0]:
            sentence = ''
            if child.tag == "terminals":
                for elem in child:
                    sentence += elem.attrib['word'] + ' '
                    #UNCOMMENT for format for files (no spaces before/after punctuation)
                    #if '$' in elem.attrib['pos']:
                        #sentence = sentence[:-1] + elem.attrib['word'] + ' '
                    #else:
                        #sentence += elem.attrib['word'] + ' '

            if sentence != '':
                all_sentences.append(sentence)

    return all_sentences[1:]

def get_gold_sentences(filename):

    with open(filename, 'r') as myfile:
        f = myfile.readlines()

    doc_sents = [line.strip() for line in f if line != '\n']

    return doc_sents

def match_edus_to_sent(article_name, path): #e.g. maz-2316

    if "PotsdamCommentaryCorpus" in path:
        sents = get_gold_syntax_sentences(path + 'syntax/' + article_name + '.xml')
    else:
        sents = get_gold_sentences(path + 'syntax/' + article_name + ".txt")

    edus, ids = get_edus(path + 'rst/' + article_name + '.rs3')

    sent_to_edu = {}
    for i, sent in enumerate(sents):
        sent_to_edu[i] = []
        for edu in edus:
            if edu in sent:
                sent_to_edu[i].append(ids[edu])
            elif sent in edu:
                sent_to_edu[i].append(ids[edu])
            elif SM(None,edu, sent).ratio() > 0.95:
                sent_to_edu[i].append(ids[edu])

    return sent_to_edu, sents
