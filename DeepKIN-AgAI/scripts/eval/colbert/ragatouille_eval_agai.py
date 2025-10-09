from deepkin.utils.misc_functions import read_lines

from ragatouille import RAGPretrainedModel

def eval_ragatouille_colbert():
    keyword = 'agai-colbert-10000'
    print(f'Evaluating {keyword} ...', flush=True)
    qa_query_id = 'kinya-ag-retrieval/rw_ag_retrieval_query_id.txt'
    qa_query_text = 'kinya-ag-retrieval/rw_ag_retrieval_query_text.txt'

    all_queries = {idx: txt for idx, txt in zip(read_lines(qa_query_id), read_lines(qa_query_text))}

    print(f'Got: {len(all_queries)} queries', flush=True)

    RAG = RAGPretrainedModel.from_index(f'ragatouille-kinya-colbert/indexes/agai-colbert-10000/')

    dev_triples = 'kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_dev.tsv'
    test_triples = 'kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_test.tsv'

    EVAL_SETS = [('DEV', dev_triples),
                 ('TEST', test_triples)]

    for eval_set_name, eval_qpn_triples in EVAL_SETS:
        eval_query_to_passage_ids = {(line.split('\t')[0]): (line.split('\t')[1]) for line in
                                     read_lines(eval_qpn_triples)}
        Top = [1, 5, 10, 20, 30]
        TopAcc = [0.0 for _ in Top]
        MTop = [5, 10, 20, 30]
        MRR = [0.0 for _ in MTop]
        Total = 0.0
        for itr, (query_id, target_doc_id) in enumerate(eval_query_to_passage_ids.items()):
            query = all_queries[query_id]
            results = RAG.search(query=query, k=max(max(50, max(Top)), max(MTop)))
            results = [(d['score'],d['document_id']) for d in results]
            Total += 1.0
            results = sorted(results, key=lambda x: x[0], reverse=True)
            for i, t in enumerate(Top):
                TopAcc[i] += (1.0 if (target_doc_id in {idx for sc, idx in results[:t]}) else 0.0)
            for i, t in enumerate(MTop):
                top_rr = [(1 / (i + 1)) for i, (sc, idx) in enumerate(results[:t]) if idx == target_doc_id]
                MRR[i] += (top_rr[0] if (len(top_rr) > 0) else 0.0)
        print(f'-------------------------------------------------------------------------------------------------')
        for i, t in enumerate(Top):
            print(f'@{eval_set_name} Final {keyword} kinya-ag-retrieval {eval_set_name} Set Top#{t} Accuracy:',
                  f'{(100.0 * TopAcc[i] / Total): .1f}% ({TopAcc[i]:.0f} / {Total:.0f})')
        for i, t in enumerate(MTop):
            print(f'@{eval_set_name} Final {keyword} kinya-ag-retrieval {eval_set_name} Set MRR@{t}:',
                  f'{(100.0 * MRR[i] / Total): .1f}% ({MRR[i]:.0f} / {Total:.0f})')
        print(f'-------------------------------------------------------------------------------------------------',
              flush=True)

if __name__ == '__main__':
    eval_ragatouille_colbert()
