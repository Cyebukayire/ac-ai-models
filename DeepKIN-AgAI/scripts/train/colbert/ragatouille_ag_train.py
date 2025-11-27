from ragatouille import RAGTrainer

from deepkin.utils.misc_functions import time_now

# This is not yet working due to BUG In RAGatouille:
# https://github.com/AnswerDotAI/RAGatouille/issues/275

if __name__ == "__main__":
    DATA_DIR = '/home/ubuntu/DATA'
    print(time_now(), 'Reading data ...', flush=True)
    with open(f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_answers.tsv', 'r', encoding='utf-8') as f:
        answers = [l.rstrip('\n').split('\t') for l in f.readlines()]
    answers = {tks[0]:tks[1] for tks in answers}

    with open(f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_questions.tsv', 'r', encoding='utf-8') as f:
        questions = [l.rstrip('\n').split('\t') for l in f.readlines()]
    questions = {tks[0]:tks[1] for tks in questions}

    with open(f'{DATA_DIR}/kinya-ag-retrieval/rw_ag_retrieval_qpntriplets_train.tsv', 'r', encoding='utf-8') as f:
        triplets = [l.rstrip('\n').split('\t') for l in f.readlines()]

    triplets = [(questions[tks[0]], answers[tks[1]], answers[tks[2]]) for tks in triplets]

    trainer: RAGTrainer = RAGTrainer(model_name="AfroColBERT", pretrained_model_name="Davlan/bert-base-multilingual-cased-finetuned-kinyarwanda", language_code="rw")

    print(time_now(), 'Preparing data ...', flush=True)

    trainer.prepare_training_data(raw_data=triplets, data_out_path=f'{DATA_DIR}/new-ragatouille-kinya-colbert/data/', mine_hard_negatives=False)

    print(time_now(), 'Training AfroColBERT ...', flush=True)

    trainer.train(batch_size=32,
                  nbits=8,  # How many bits will the trained model use when compressing indexes
                  maxsteps=500000,  # Maximum steps hard stop
                  use_ib_negatives=True,  # Use in-batch negative to calculate loss
                  dim=512,  # How many dimensions per embedding. 128 is the default and works well.
                  learning_rate=5e-6,
                  # Learning rate, small values ([3e-6,3e-5] work best if the base model is BERT-like, 5e-6 is often the sweet spot)
                  doc_maxlen=512,
                  # Maximum document length. Because of how ColBERT works, smaller chunks (128-256) work very well.
                  use_relu=False,  # Disable ReLU -- doesn't improve performance
                  warmup_steps="auto",  # Defaults to 10%
                  )

    print(time_now(), 'DONE Training AfroColBERT!', flush=True)



