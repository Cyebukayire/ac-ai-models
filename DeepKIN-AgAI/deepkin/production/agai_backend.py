from __future__ import print_function, division

import glob
import os.path
import traceback
# Ignore warnings
import warnings
from typing import Tuple, List

import progressbar
import torch
import torch.nn.functional as F
from deepkin.clib.libkinlp.kinlpy import parse_text_to_morpho_sentence, ParsedFlexSentence
from deepkin.data.morpho_qa_triple_data import DOCUMENT_TYPE_ID, QUESTION_TYPE_ID
from deepkin.inference.qaret_inference import init_qaret_inference_setup
from deepkin.models.morpho_bert import KinyaColBERT
from deepkin.utils.misc_functions import time_now, read_lines

warnings.filterwarnings("ignore")

from flask import Flask, request

agai_model_setup = None

documents_data = dict()
AGAI_DATA_DIR = 'KINLP/data/agai'

Doc_Mean = None
Doc_Stdev = None
Query_Mean = None
Query_Stdev = None

def aggregate_embeddings():
    global agai_model_setup
    global Doc_Mean
    global Doc_Stdev
    global Query_Mean
    global Query_Stdev

    print(time_now(), 'Aggregating embeddings...', flush=True)

    (model, device, lib, ffi) = agai_model_setup
    model.eval()

    qa_query_id = 'KINLP/data/AgQA_2025-05-22/AgQA_query_id_2025-05-22.txt'
    qa_query_text = 'KINLP/data/AgQA_2025-05-22/parsed_AgQA_query_text_2025-05-22.txt'
    qa_passage_id = 'KINLP/data/AgQA_2025-05-22/AgQA_passage_id_2025-05-22.txt'
    qa_passage_text = 'KINLP/data/AgQA_2025-05-22/parsed_AgQA_passage_text_2025-05-22.txt'

    all_queries = {idx: ParsedFlexSentence(txt).trim(508) for idx, txt in zip(read_lines(qa_query_id),
                                                              read_lines(qa_query_text))}
    all_passages = {idx: ParsedFlexSentence(txt).trim(508) for idx, txt in zip(read_lines(qa_passage_id),
                                                              read_lines(qa_passage_text))}

    print(f'Got: {len(all_queries)} queries, {len(all_passages)} passages', flush=True)

    DocPool = []
    QueryPool = []
    with torch.no_grad():
        print(time_now(), 'Embedding passages ...', flush=True)
        with progressbar.ProgressBar(max_value=len(all_passages), redirect_stdout=True, redirect_stderr=True, initial_value=0) as bar:
            for itr,(passage_id,passage) in enumerate(all_passages.items()):
                if (itr % 100) == 0:
                    bar.update(itr)
                D = model.get_colbert_embeddings([passage], DOCUMENT_TYPE_ID)
                DocPool.append(D.view(-1,D.size(-1)))# if DocPool is None else torch.cat((DocPool, D.view(-1,D.size(-1))))

        Doc_Mean = torch.cat(DocPool).mean(dim=0)
        Doc_Stdev = torch.cat(DocPool).std(dim=0)
        del DocPool

        print(time_now(), 'Embedding queries ...', flush=True)
        with progressbar.ProgressBar(max_value=len(all_queries), redirect_stdout=True, redirect_stderr=True, initial_value=0) as bar:
            for itr,(_,qr) in enumerate(all_queries.items()):
                if (itr % 1000) == 0:
                    bar.update(itr)
                Q = model.get_colbert_embeddings([qr], QUESTION_TYPE_ID)
                QueryPool.append(Q.view(-1, Q.size(-1))) #if QueryPool is None else torch.cat((QueryPool, Q.view(-1, Q.size(-1))))

        Query_Mean = torch.cat(QueryPool).mean(dim=0)
        Query_Stdev = torch.cat(QueryPool).std(dim=0)
        del QueryPool

        print(time_now(), 'Aggregation complete!', flush=True)


def agai_setup(rank=0,
               sock_file="KINLP/data/run/deepkin.sock"):
    global agai_model_setup
    global documents_data

    AgQA_model_file = 'KINLP/data/agai_real_morphobert_large_qaret_512_eval_2025-05-27.pt_14.8K.pt'

    agai_model_setup = init_qaret_inference_setup(AgQA_model_file, rank=rank, sock_file = sock_file)
    (AgQA_ColBERT, device, lib, ffi) = agai_model_setup
    documents_data = dict()
    for fn in glob.glob(f"{AGAI_DATA_DIR}/content/*.text.txt"):
        id = fn.split('/')[-1].split('.')[0]
        with open(fn, 'r', encoding='utf-8') as file:
            text = file.read().rstrip()
        embeddings = None

        if os.path.isfile(f"{AGAI_DATA_DIR}/embeddings/{id}.embeddings.pt"):
            embeddings = torch.load(f"{AGAI_DATA_DIR}/embeddings/{id}.embeddings.pt", weights_only=True).to(device)

        documents_data[id] = (text, embeddings)
    print(f'Got {len(documents_data)} documents!', flush=True)
    aggregate_embeddings()
    return agai_model_setup, documents_data


def save_doc(id:str, text:str, example_questions: List[str]) -> bool:
    global agai_model_setup
    global documents_data
    (AgQA_ColBERT, device, lib, ffi) = agai_model_setup
    AgQA_ColBERT.eval()
    try:
        doc_encoder: KinyaColBERT = AgQA_ColBERT
        embeddings = doc_encoder.get_colbert_embeddings([parse_text_to_morpho_sentence(ffi, lib, text).trim(508)], DOCUMENT_TYPE_ID)
        with open(f"{AGAI_DATA_DIR}/content/{id}.text.txt", 'w', encoding='utf-8') as file:
            file.write(text)
        torch.save(embeddings, f"{AGAI_DATA_DIR}/embeddings/{id}.embeddings.pt")
        documents_data[id] = (text, embeddings)
        return True
    except Exception as e:
        print("An error occurred while indexing document:", e, flush=True)
        stack_trace = traceback.format_exc()
        print(stack_trace, flush=True)
        return False

def query_qaret(query_text:str, n: int) -> List[Tuple[str,str,float,float]]:
    global agai_model_setup
    global documents_data
    global Doc_Mean
    global Doc_Stdev
    global Query_Mean
    global Query_Stdev
    (AgQA_ColBERT, device, lib, ffi) = agai_model_setup
    AgQA_ColBERT.eval()
    ret = []
    try:
        query_encoder: KinyaColBERT = AgQA_ColBERT
        Q = query_encoder.get_colbert_embeddings([parse_text_to_morpho_sentence(ffi, lib, query_text).trim(508)], QUESTION_TYPE_ID)
        Q = (Q - Query_Mean) / Query_Stdev
        Q = F.normalize(Q, p=2, dim=2)
        doc_scores = []
        for doc_id in documents_data.keys():
            (doc_text, D) = documents_data[doc_id]
            D = (D - Doc_Mean) / Doc_Stdev
            D = F.normalize(D, p=2, dim=2)
            score = query_encoder.pairwise_score(Q, D).squeeze().item()
            doc_scores.append((doc_id, score/Q.size(1)))

        doc_scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = doc_scores[:n]

        for i,(doc_id,doc_score) in enumerate(top_docs):
            doc_text = documents_data[doc_id][0]
            ret.append((doc_id, doc_text, doc_score, 0.0))
        return ret
    except Exception as e:
        print(f"An error occurred while querying '{query_text}' [{n}]:", e, flush=True)
        stack_trace = traceback.format_exc()
        print(stack_trace, flush=True)
        return ret

def query(query_text:str, n: int) -> List[Tuple[str,str,float,float]]:
    return query_qaret(query_text, n)

agai_setup()
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
app.config["JSONIFY_MIMETYPE"] = "application/json; charset=utf-8"

@app.route('/save_doc', methods=['POST'])
def save_doc_task():
    content_type = request.headers.get('Content-Type')
    if ('application/json' in content_type):
        json = request.get_json()
        id = json['id']
        text = json['text']
        example_questions = json['example_questions']
        with torch.no_grad():
            output = save_doc(id, text, example_questions)
        json['result'] = output
        json['task'] = 'save_doc'
        return json
    else:
        return 'Content-Type not supported!'

@app.route('/query', methods=['POST'])
def query_task():
    content_type = request.headers.get('Content-Type')
    if ('application/json' in content_type):
        json = request.get_json()
        # print(time_now(), 'Agai Query:', json, flush=True)
        text = json['text']
        n = json['n']
        with torch.no_grad():
            ret = query(text, int(n))
        result = []
        for (doc_id,doc_text,doc_score,doc_prob) in ret:
            val = {'id':doc_id, 'text':doc_text, 'score':doc_score, 'prob':doc_prob}
            result.append(val)
        json['result'] = result
        json['task'] = 'query'
        # print(time_now(), 'Agai Response:', json, flush=True)
        return json
    else:
        return 'Content-Type not supported!'

if __name__ == '__main__':
    # app.run(debug=False, host='0.0.0.0', port=9697)
    app.run()
