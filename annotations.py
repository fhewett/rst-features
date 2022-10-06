import json

def extract_anno(filename, path):
    """Uses the JSON summaries file to output a dict with the keys as sentences and the values the importance annotation"""

    annotations = json.load(open(path + 'pcc-summaries/corpus.json', 'r'))

    d = annotations[filename]
    anno_id = {sent_id: d[sent_text] for sent_id,sent_text in enumerate(d)}

    return anno_id
