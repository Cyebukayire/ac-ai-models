import sys

import progressbar
import torch
import torch.nn.functional as F

from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence
from deepkin.data.morpho_qa_triple_data import DOCUMENT_TYPE_ID, QUESTION_TYPE_ID
from deepkin.models.kinyabert import KinyaColBERT
from deepkin.utils.misc_functions import read_lines

def eval_kinya_col_bert(DATA_DIR: str, rank = 0):
    pretrained_model_file = f'{DATA_DIR}/kinya_colbert_large_rw_ag_retrieval_finetuned_512D.pt'
    keyword = f'kinya_colbert_large'

    qa_query_id = f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_query_id.txt'
    qa_query_text = f'{DATA_DIR}/kinya-ag-retrieval/parsed_rw_ag_retrieval_query_text.txt'
    qa_passage_id = f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_passage_id.txt'
    qa_passage_text = f'{DATA_DIR}/kinya-ag-retrieval/parsed_rw_ag_retrieval_passage_text.txt'

    all_queries = {idx: ParsedFlexSentence(txt) for idx, txt in zip(read_lines(qa_query_id), read_lines(qa_query_text))}
    all_passages = {idx: ParsedFlexSentence(txt) for idx, txt in zip(read_lines(qa_passage_id), read_lines(qa_passage_text))}

    print(f'Got: {len(all_queries)} queries, {len(all_passages)} passages', flush=True)

    device = torch.device('cuda:%d' % rank)

    model, args = KinyaColBERT.from_pretrained(device, pretrained_model_file, ret_args=True)
    model.float()
    model.eval()

    passage_embeddings = dict()
    DocPool = None
    QueryPool = None
    with torch.no_grad():
        print(f'{keyword} Embedding passages ...', flush=True)
        with progressbar.ProgressBar(max_value=len(all_passages), redirect_stdout=True) as bar:
            for itr, (passage_id, passage) in enumerate(all_passages.items()):
                if (itr % 100) == 0:
                    bar.update(itr)
                passage.trim(508)
                with torch.no_grad():
                    D = model.get_colbert_embeddings([passage], DOCUMENT_TYPE_ID)
                DocPool = D.view(-1,D.size(-1)) if DocPool is None else torch.cat((DocPool, D.view(-1,D.size(-1))))
                passage_embeddings[passage_id] = D

        query_embeddings = dict()
        Doc_Mean = DocPool.mean(dim=0)
        Doc_Stdev = DocPool.std(dim=0)
        del DocPool
        print(f'{keyword} Embedding queries ...', flush=True)
        with progressbar.ProgressBar(max_value=len(all_queries), redirect_stdout=True) as bar:
            for itr, (query_id, query) in enumerate(all_queries.items()):
                if (itr % 1000) == 0:
                    bar.update(itr)
                query.trim(508)
                with torch.no_grad():
                    Q = model.get_colbert_embeddings([query], QUESTION_TYPE_ID)
                QueryPool = Q.view(-1, Q.size(-1)) if QueryPool is None else torch.cat((QueryPool, Q.view(-1, Q.size(-1))))
                query_embeddings[query_id] = Q

        Query_Mean = QueryPool.mean(dim=0)
        Query_Stdev = QueryPool.std(dim=0)
        del QueryPool

        dev_triples = f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_dev.tsv'
        test_triples = f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_test.tsv'

        EVAL_SETS = [('DEV', dev_triples),
                     ('TEST', test_triples)]

    for eval_set_name, eval_qpn_triples in EVAL_SETS:
        eval_query_to_passage_ids = {(line.split('\t')[0]): (line.split('\t')[1]) for line in read_lines(eval_qpn_triples)}
        Top = [1, 5, 10, 20, 30]
        TopAcc = [0.0 for _ in Top]
        MTop = [5, 10, 20, 30]
        MRR = [0.0 for _ in MTop]
        Total = 0.0
        for itr,(query_id,target_doc_id) in enumerate(eval_query_to_passage_ids.items()):
            query = all_queries[query_id]
            with torch.no_grad():
                Q = model.get_colbert_embeddings([query], QUESTION_TYPE_ID)
            Q = (Q - Query_Mean) / Query_Stdev
            Q = F.normalize(Q, p=2, dim=2)
            results = []
            for doc_id,D in passage_embeddings.items():
                D = (D - Doc_Mean) / Doc_Stdev
                D = F.normalize(D, p=2, dim=2)
                with torch.no_grad():
                    score = model.pairwise_score(Q,D).squeeze().item()
                score = score / Q.size(1)
                results.append((score, doc_id))
            Total += 1.0
            results = sorted(results, key=lambda x: x[0], reverse=True)
            for i, t in enumerate(Top):
                TopAcc[i] += (1.0 if (target_doc_id in {idx for sc, idx in results[:t]}) else 0.0)
            for i, t in enumerate(MTop):
                top_rr = [(1 / (i + 1)) for i, (sc, idx) in enumerate(results[:t]) if idx == target_doc_id]
                MRR[i] += (top_rr[0] if (len(top_rr) > 0) else 0.0)
        print(f'-------------------------------------------------------------------------------------------------')
        for i, t in enumerate(Top):
            print(f'@{eval_set_name} Final {keyword}-{args.colbert_embedding_dim} kinya-ag-retrieval {eval_set_name} Set Top#{t} Accuracy:',
                  f'{(100.0 * TopAcc[i] / Total): .1f}% ({TopAcc[i]:.0f} / {Total:.0f})')
        for i, t in enumerate(MTop):
            print(f'@{eval_set_name} Final {keyword}-{args.colbert_embedding_dim} kinya-ag-retrieval {eval_set_name} Set MRR@{t}:',
                  f'{(100.0 * MRR[i] / Total): .1f}% ({MRR[i]:.0f} / {Total:.0f})')
        print(f'-------------------------------------------------------------------------------------------------', flush=True)

if __name__ == '__main__':
    DATA_DIR = sys.argv[1] # '/home/ubuntu/DATA'
    eval_kinya_col_bert(DATA_DIR, rank = 0)
