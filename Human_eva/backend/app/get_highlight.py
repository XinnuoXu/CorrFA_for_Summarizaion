#%%
"""
Extract results from database and then perform analysis on top of to draw insights
"""
import json
import math
import os, sys
from nltk.util import ngrams
from nltk.stem.porter import *
from collections import Counter
from itertools import chain
import pandas as pd
from flask_sqlalchemy import SQLAlchemy

sys.path.append(os.path.abspath('../../'))
from backend.models import Dataset, Document, Summary, SummaryGroup, AnnotationResult, DocStatus
from backend.app import create_app

DELETE_UNVALID_DATA=False

#%%
# Loading data from database
app = create_app()
db = SQLAlchemy(app)
#q_results = db.session.query(Document, Dataset) \
#    .join(Dataset) \
#    .filter(Dataset.name == 'BBC').all()

#%%
# Process data from database into components and components' type
def parse(doc_json):
    """
    Parse document into components (list of all tokens) and comp_types (list of types for all tokens)
    """
    components = []
    comp_types = []
    index = []
    for sent in doc_json['sentences']:
        for idx, token in enumerate(sent['tokens']):
            aWord = token['word'].lower()
            if token['word'] == '-LRB-':
                aWord = '('
            elif token['word'] == '-RRB-':
                aWord = ')'
            elif token['word'] == '``':
                aWord = '"'
            elif token['word'] == '\'\'':
                aWord = '"'
            components.append(aWord)
            if aWord.strip() == '':
                comp_types.append('whitespace')
            else:
                comp_types.append('word')
            index.append(len(index))
            if idx != len(sent['tokens']) - 2:
                components.append(' ')
                comp_types.append('whitespace')
                index.append(len(index))
    data = {
        'content': pd.Series(components),
        'type': pd.Series(comp_types),
        'index': pd.Series(index),
    }
    return pd.DataFrame(data)


# Contains information of each word in the document
q_results = db.session.query(Document, Dataset) \
    .join(Dataset).all()

df_doc_prop = pd.DataFrame([])
summaries = {}; rec_id_2_doc_id = {}
for doc, _ in q_results:
    doc_json = json.loads(doc.doc_json)
    df_doc_prop = df_doc_prop.append(parse(doc_json)\
        .assign(doc_id=doc_json['doc_id'])\
        .assign(rec_id=doc.id))
    summaries[doc.id] = doc.summary
    rec_id_2_doc_id[doc.id] = doc_json['doc_id']


# Contains data of the document with the summary
df_doc = pd.DataFrame(df_doc_prop[df_doc_prop['type'] != 'whitespace'].groupby('rec_id').count())
df_doc = df_doc.rename(columns={'index': 'word_len'}).drop(columns=['type', 'content'])
df_doc['summ'] = df_doc.index.map(summaries.get)

#%%
# Retrieve highlights
def process_doc(results, word_idx, doc_id):
    """
    Build indexes and texts for the given document
    """
    indexes = []
    texts = []
    result_ids = []
    for result_id, data in results:
        for h_id, highlight in data.items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'].lower())
            result_ids.append(result_id)
    data = {
        'indexes': pd.Series(indexes),
        'text': pd.Series(texts),
        'result_id': pd.Series(result_ids),
        'doc_id': doc_id
    }
    return pd.DataFrame(data)

Ann_status_id_2_doc_id = {}
doc_status = db.session.query(DocStatus)
for ds in doc_status:
    Ann_status_id_2_doc_id[ds.id] = ds.doc_id

annotation_dicts = {}
annotation_res = db.session.query(AnnotationResult)
for ann in annotation_res:
    try:
        result_json = json.loads(ann.result_json)
    except:
        continue
    if 'highlights' not in result_json or len(result_json['highlights']) == 0:
        continue
    result_json = result_json['highlights']
    result_id = ann.id
    status_id = ann.status_id
    if DELETE_UNVALID_DATA and ann.validity != 1:
        continue
    if status_id not in Ann_status_id_2_doc_id:
        continue
    rec_id = Ann_status_id_2_doc_id[status_id]
    if rec_id not in annotation_dicts:
        annotation_dicts[rec_id] = []
    annotation_dicts[rec_id].append((result_id, result_json))

df_h = pd.DataFrame([])
for rec_id in annotation_dicts:
    word_idx = df_doc_prop.groupby('rec_id').get_group(rec_id)
    pdoc = process_doc(annotation_dicts[rec_id], word_idx, rec_id_2_doc_id[rec_id])
    if pdoc is not None:
        df_h = df_h.append(pdoc.assign(rec_id=rec_id))

#%%
# Retrieve result per coder and store them in DataFrame
q_results = db.session.query(Document).all()

df_annotations = pd.DataFrame([])
df_doc_prop_group = df_doc_prop.groupby('rec_id')
count = 0
test = {}
for doc in q_results:
    if doc.id not in annotation_dicts:
        continue
    results = annotation_dicts[doc.id]
    for result_id, result in results:
        indexes = []
        texts = []
        word_idx = df_doc_prop_group.get_group(doc.id)
        for key, highlight in result.items():
            if highlight['text'] == '':
                continue
            word_only_highlight = [idx for idx in highlight['indexes'] if word_idx.loc[idx]['type'] == 'word']
            indexes.append(word_only_highlight)
            texts.append(highlight['text'])
        df_annotations = df_annotations.append(
            pd.DataFrame({
                'indexes': pd.Series(indexes),
                'texts': pd.Series(texts)
            }).assign(doc_id=doc.doc_id, result_id=result_id, rec_id=doc.id))
#%%
# Modified n-gram
df_doc = df_doc.assign(doc_text=lambda x: df_doc_prop[df_doc_prop['type'] == 'word'].groupby('rec_id')['content'].apply(list))

df_doc = df_doc.assign(doc_text_join=lambda x: df_doc['doc_text'].apply(' '.join))

df_doc = df_doc.assign(h_idxs=lambda x: df_h.groupby('rec_id').apply(lambda x: list(set(chain(*x.indexes)))))

for i, h_idx in enumerate(df_doc['h_idxs']):
    if not isinstance(h_idx, list):
        df_doc['h_idxs'][i+1] = []

df_doc = df_doc.assign(doc_idxs=lambda x: df_doc_prop[df_doc_prop['type']=='word'].groupby('rec_id')['index'].apply(list))

df_doc = df_doc.assign(h_len=lambda x: df_doc[['h_idxs']].apply(lambda x: len(x['h_idxs']), axis=1))

MAX_LEN = 50
df_ngrams = pd.DataFrame([])

#%%
def numH(w, H):
    result = 0
    H_group = H.groupby('result_id')
    highlights = {}
    result_ids = set()
    for result_id, data in H_group:
        result_ids.add(result_id)
        if result_id not in highlights.keys():
            highlights[result_id] = data['indexes']
        else:
            highlights[result_id].append(data['indexes'])
    for result_id in highlights.keys():
        h_words = list(chain(*[highlight for highlight in highlights[result_id]]))
        if w in h_words:
            result += len(h_words) / MAX_LEN
    return result, len(result_ids)

def beta(n, D, H):
    m = len(D[0])
    score_list = []
    phrase_list = []
    idx_dict = {}
    for i in range(m-n+1):
        phrase = " ".join(D[0][i:i+n])
        if phrase not in idx_dict:
            idx_dict[phrase] = 0
        else:
            idx_dict[phrase] += 1
        phrase_list.append(str(idx_dict[phrase]) + "-" + phrase)
        total_NumH = 0
        for j in range(i, i+n):
            numh, res_num = numH(D[1][j], H)
            total_NumH += numh / (res_num * n)
        score_list.append(total_NumH)
    return phrase_list, score_list

df_h_g = df_h.groupby('rec_id')
fpout = open("highlight.jsonl", "w")
for rec_id, data in df_annotations.groupby('rec_id'):
    summ = summaries[rec_id]
    doc_texts = (list(df_doc.loc[rec_id]['doc_text']), list(df_doc.loc[rec_id]['doc_idxs']))
    H = df_h_g.get_group(rec_id)

    json_obj = {}
    json_obj["record_id"] = rec_id
    json_obj["doc_id"] = json.loads(data.doc_id.to_json(orient='split'))["data"][0]
    json_obj["document"] = doc_texts[0]
    json_obj["summary"] = json.loads(summ)
    print (H)
    json_obj["uni_gram"], json_obj["uni_gram_scores"] = beta(1, doc_texts, H)
    json_obj["bi_gram"], json_obj["bi_gram_scores"] = beta(2, doc_texts, H)
    json_obj["tri_gram"], json_obj["tri_gram_scores"] = beta(3, doc_texts, H)
    json_obj["qua_gram"], json_obj["qua_gram_scores"] = beta(4, doc_texts, H)
    fpout.write(json.dumps(json_obj) + "\n")
fpout.close()
