import re

from initialise import get_root_node_rst, find_path

def initialise_depth(filename):
    """Takes the rs3 filename as input
        init_dict : dictionary where keys are parent IDs, values are segment/group IDs
        nuc_init_dict : same as above but only with nuclear relations
        all_seg_ids : a list of all EDU level segment IDs
        satellites: list of IDs for relations which are not span or MN
        inner_nodes : list of all non-terminal nodes
        nuc_seg_ids : a list of nuclear terminal IDs"""


    init_dict, nuc_init_dict = dict(), dict()
    all_seg_ids, satellites, inner_nodes, nuc_seg_ids = list(), list(), list(), list()

    with open(filename, 'r') as myfile:
        read_file = myfile.read()

        group_id = r'<group id="([0-9]*)" type="(\S*?)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_group = re.compile(group_id)
        for (groupid, typetype, parent, relname) in re.findall(regex_group, read_file):
            inner_nodes.append(int(groupid))

            if int(parent) not in init_dict:
                init_dict[int(parent)] = []
            init_dict[int(parent)].append(int(groupid))

            if relname in ["conjunction", "contrast", "disjunction", "joint",
                        "list", "restatement-mn", "sequence", "span"]:
                nuc_seg_ids.append(int(groupid))
                if int(parent) not in nuc_init_dict:
                    nuc_init_dict[int(parent)] = []
                nuc_init_dict[int(parent)].append(int(groupid))

            else:
                satellites.append(int(groupid))


        regex5 = r'<segment id="([0-9]*)" parent="([0-9]*)" relname="(\S*?)"[\S ]*'
        regex_seg = re.compile(regex5)
        for (segid, segparent, segrelname) in re.findall(regex_seg, read_file):
            if int(segparent) not in init_dict:
                init_dict[int(segparent)] = []
            init_dict[int(segparent)].append(int(segid))
            if int(segid) not in init_dict:
                init_dict[int(segid)] = [int(segid)]
            all_seg_ids.append(int(segid))

            if segrelname in ["conjunction", "contrast", "disjunction", "joint",
                               "list", "restatement-mn", "sequence", "span"]:
                nuc_seg_ids.append(int(segid))
                if int(segparent) not in nuc_init_dict:
                    nuc_init_dict[int(segparent)] = []
                nuc_init_dict[int(segparent)].append(int(segid))
                if int(segid) not in nuc_init_dict:
                    nuc_init_dict[int(segid)] = [int(segid)]
            else:
                satellites.append(int(segid))


    return init_dict, all_seg_ids, satellites, nuc_init_dict, inner_nodes

def get_paths(all_seg_ids, init_dict, parent):
    """Gets paths from root node to every leaf node using all inner nodes (i.e. both nucs and sats)"""

    paths = []
    for x in all_seg_ids:
        paths.append(find_path(init_dict, parent, x))

    return paths


def get_node_depths(paths, satellites):

    node_depth = {}

    #how many numbers are infront of it, length not including satellite relations
    for path in paths:
        for i, segid in enumerate(path):
            minus_points = 0
            #print(path[:i+1])
            for elem in path[:i+1]:
                if elem in satellites:
                    minus_points += 1
                node_depth[segid] = len(path[:i]) - minus_points

    return node_depth

def nuclear_paths(root_node, pcc_path, fn):

    _, all_segs, _, nuc_id, inners = initialise_depth(pcc_path + fn)
    inners.append(root_node)

    nuc_paths = []
    for start in inners:
        for x in all_segs:
            ppp = find_path(nuc_id, start, x)
            if ppp != None:
                nuc_paths.append(ppp)

    return nuc_paths

def get_final_depth_values(nuc_paths, node_depth, sats, all_segs):

    segment_depth = {}
    for p in nuc_paths:
        segment_depth[p[-1]] = node_depth[p[0]]

    sats_depth = {}
    for seg in segment_depth:
        if seg in sats:
            current_depth = node_depth[seg]
            sats_depth[sats[seg]] = current_depth

    almost_all_depth = {**sats_depth, **segment_depth}

    newbies = {}
    for seg in all_segs:
        if seg not in almost_all_depth:
            newbies[seg] = node_depth[seg]

    final_depth = {**almost_all_depth, **newbies}

    return final_depth

def get_depth_scores(filename, path):

    try:

        init_dict1, all_seg_ids1, satellites1, nuc_init_dict1, inner_nodes1 = initialise_depth(path + 'rst/' + filename)
        parent1 = get_root_node_rst(path + 'rst/' + filename)
        paths1 = get_paths(all_seg_ids1, init_dict1, parent1)
        node_depths1 = get_node_depths(paths1, satellites1)
        nuc_paths1 = nuclear_paths(parent1, path + 'rst/', filename)
        final_depth1 = get_final_depth_values(nuc_paths1, node_depths1, satellites1, all_seg_ids1)
        total_depth = max(list(final_depth1.values())) + 1

        scores = dict()
        for terminal_node in final_depth1:
            scores[terminal_node] = (total_depth - final_depth1[terminal_node]) / total_depth


    except TypeError:
        print("TypeError whilst calculating the depth score for ", filename)
        return scores

    return scores
