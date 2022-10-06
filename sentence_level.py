import re

from initialise import get_root_node_rst, match_edus_to_sent, find_path

def helper_sent(filename):
    """Takes the rs3 filename as input
        init_dict : dictionary where keys are parent IDs, values are segment/group IDs
        outgoing :  dict with keys which are child IDs, vals are relations
        incoming : dict with keys which are parent IDs, vals are relations"""


    init_dict, outgoing, incoming = dict(), dict(), dict()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()

        group_id = r'<group id="([0-9]*)" type="(\S*?)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_group = re.compile(group_id)
        for (groupid, typetype, parent, relname) in re.findall(regex_group, read_file):
            if int(parent) not in init_dict:
                init_dict[int(parent)] = []
            init_dict[int(parent)].append(int(groupid))

            if int(groupid) not in outgoing:
                outgoing[int(groupid)] = relname
            else: #if it is there, only overwrite if it's span or MN
                if outgoing[int(groupid)] in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence", "span"]:
                    outgoing[int(groupid)] = relname
                else:
                    continue

            if int(parent) not in incoming:
                incoming[int(parent)] = relname
            else: #if it is there, only overwrite if it's span
                if incoming[int(parent)] in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence", "span"]:
                    incoming[int(parent)] = relname
                else:
                    continue

        regex5 = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_seg = re.compile(regex5)
        for (segid, segparent, segrelname) in re.findall(regex_seg, read_file):

            if int(segparent) not in init_dict:
                init_dict[int(segparent)] = []
            init_dict[int(segparent)].append(int(segid))
            if int(segid) not in init_dict:
                init_dict[int(segid)] = [int(segid)]

            if int(segid) not in outgoing:
                outgoing[int(segid)] = segrelname
            else: #if it is there, only overwrite if it's span or multinuc???
                if outgoing[int(segid)] in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence", "span"]:
                    outgoing[int(segid)] = segrelname
                else:
                    continue

            if int(segparent) not in incoming:
                incoming[int(segparent)] = segrelname
            else: #if it is there, only overwrite if it's span or MN
                if incoming[int(segparent)] in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence", "span"]:
                    incoming[int(segparent)] = segrelname
                else:
                    continue

    return init_dict, outgoing, incoming


### Step 1, get relevant EDUs (where an EDU is less than a sentence)
def get_sent_edus(filename, path):

    sent_ex, _ = match_edus_to_sent(filename, path)

    sent_edu_ids, sent_ids = list(), list()

    for k,v in sent_ex.items():
        if len(v) > 1:
            sent_edu_ids.append(v)
            sent_ids.append(k)

    return sent_edu_ids, sent_ids

def get_sent_level(fn, path):

    sent_level = dict()

    edus_to_check, sentids = get_sent_edus(fn, path)

    root =  get_root_node_rst(path + "rst/" + fn + ".rs3")

    ini,outgoing,incoming = helper_sent(path + 'rst/' + fn + '.rs3')

    ### Step 2, get path from root node to these EDUs, find longest path, get the value which is
    #the number of EDUs away from the end

    for i, edus in enumerate(edus_to_check):
        all_paths = dict()
        for edu in edus:
            path = find_path(ini, root, edu)
            all_paths[len(path)] = path

        if len(all_paths) == 1: #MN exception
            node = all_paths[max(all_paths)][-len(edus)]
        else:
            node = all_paths[max(all_paths)][-(len(edus)+1)]

    ### For this value: what's the outgoing relation?
    # If it's span, then it's nucleus. What's the incoming relation? that's the relation
    # If it's a relation, then it's a satellite of this relation

        if outgoing[node] in ["conjunction", "contrast", "disjunction", "joint", "list", "restatement-mn", "sequence"]:
            sent_level[sentids[i]] = {'relations': outgoing[node], 's_or_n': 'N'}
        elif outgoing[node] == "span":
            sent_level[sentids[i]] = {'relations': incoming[node], 's_or_n': 'N'}
        else:
            sent_level[sentids[i]] = {'relations': outgoing[node], 's_or_n': 'S'}

    return sent_level
