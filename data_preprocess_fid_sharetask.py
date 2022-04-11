from datasets import load_dataset
from collections import defaultdict
import json
import random

def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()

def split_text_section(spans, title):
    def get_text(buff, title, span):
        text = " ".join(buff).replace("\n", " ")
        parent_titles = [title.replace("/", "-").rsplit("#")[0]]
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
        text = " / ".join(parent_titles) + " // " + text
        return text2line(text)

    buff = []
    pre_sec, pre_title, pre_span = None, None, None
    passages = []
    subtitles = []
    for span in spans:
        parent_titles = title
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
            parent_titles = " / ".join(parent_titles)
        if pre_sec == span["id_sec"] or pre_title == span["title"].strip():
            buff.append(span["text_sp"])
        elif buff:
            text = get_text(buff, title, pre_span)
            passages.append(text)
            subtitles.append(parent_titles)
            buff = [span["text_sp"]]
        else:
            buff.append(span["text_sp"])
        pre_sec = span["id_sec"]
        pre_span = span
        pre_title = span["title"].strip()
    if buff:
        text = get_text(buff, title, span)
        passages.append(text)
        subtitles.append(parent_titles)
    return passages, subtitles

doc_dataset = load_dataset("doc2dial_pub.py", "document_domain", split="train", ignore_verifications=True)
d_doc_data = defaultdict(dict)  # doc -> "doc_text", "spans"
d_doc_psg = {}
doc_psg_all = []
doc_titles_all = []
doc_domain_all = []
d_pid_domain = {}
start_idx = 0
for ex in doc_dataset:
    passages, subtitles = split_text_section(ex["spans"], ex["title"])
    doc_psg_all.extend(passages)
    doc_titles_all.extend(subtitles)
    doc_domain_all.extend([ex["domain"]] * len(passages))
    d_doc_psg[ex["doc_id"]] = (start_idx, len(passages))
    for i in range(start_idx, start_idx + len(passages)):
        d_pid_domain[i] = ex["domain"]
    start_idx += len(passages)
    d_doc_data[ex["doc_id"]]["doc_text"] = ex["doc_text"]
    d_doc_data[ex["doc_id"]]["spans"] = {}
    d_doc_data[ex["doc_id"]]["domain"] = ex["domain"]
    for d_span in ex["spans"]:
        d_doc_data[ex["doc_id"]]["spans"][d_span["id_sp"]] = d_span
pass

id_list = []
question_list = []
ctxs_list = []
with open('./data/mdd_share/share.id', 'r') as f:
    for line in f:
        id_list.append(line.strip())
with open('./data/mdd_share/share.source', 'r') as f:
    for line in f:
        question_list.append(line.strip())
with open('./data/retrieval_result/share.retrieval_id', 'r') as f:
    for line in f:
        number_list = [int(i) for i in line.split('\t')][:50]
        ctxs = []
        for num in number_list:
            psg_idx = num
            ctxs.append({"title":"", "text":doc_psg_all[psg_idx]})
        ctxs_list.append(ctxs)
share_data = [{'id':id, 'question':question, 'target':'', 'answers':[''], 'ctxs':ctxs} for id, question, ctxs in zip(id_list, question_list, ctxs_list)]
with open('./data/share_50.json', 'w') as f:
    json.dump(share_data, f, indent=4)

