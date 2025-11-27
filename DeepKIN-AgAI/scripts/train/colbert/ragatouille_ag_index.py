from ragatouille import RAGPretrainedModel

from deepkin.utils.misc_functions import time_now

# This is not yet working due to BUG In RAGatouille:
# https://github.com/AnswerDotAI/RAGatouille/issues/275

if __name__ == "__main__":
    DATA_DIR = '/home/ubuntu/DATA'
    print(time_now(), 'Reading data ...', flush=True)
    with open(f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_answers.tsv', 'r', encoding='utf-8') as f:
        answers = [l.rstrip('\n').split('\t') for l in f.readlines()]
    collection = [tks[1] for tks in answers]
    document_ids = [tks[0] for tks in answers]
    print(time_now(), f'Indexing ...', flush=True)
    RAG: RAGPretrainedModel = RAGPretrainedModel.from_pretrained(f"{DATA_DIR}/new-ragatouille-kinya-colbert/checkpoints/colbert-10000/")
    RAG.index(collection=collection,
              document_ids=document_ids,
              index_name="agai-colbert-10000",
              max_document_length=512,
              split_documents=False)
    print(time_now(), f'DONE with Indexing!', flush=True)
