from __future__ import print_function, division

# Ignore warnings
import warnings

import numpy as np
import progressbar
import torch
from scipy import stats
from seqeval.metrics import f1_score as tag_f1_score
from sklearn.metrics import f1_score as cls_f1_score

from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence
from deepkin.models.kinyabert import New_KinyaBERT_Classifier_from_pretrained_kinyabert
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import read_lines

warnings.filterwarnings("ignore")
# %%
def spearman_corr(r_x, r_y):
    return stats.spearmanr(r_x, r_y)[0]

def pearson_corr(r_x, r_y):
    return stats.pearsonr(r_x, r_y)[0]

if __name__ == '__main__':
    args = FlexArguments().parse_args()
    rank = args.local_rank
    device = torch.device('cuda:%d' % rank)
    inputs0_list = [ParsedFlexSentence(l) for l in read_lines(args.cls_test_input0)]
    inputs1_list = None if ((args.cls_test_input1 is None) or (len(args.cls_test_input1) == 0)) else [ParsedFlexSentence(l) for l in read_lines(args.cls_test_input1)]
    cls_model = New_KinyaBERT_Classifier_from_pretrained_kinyabert(device, args.pretrained_bert_model_file)
    cls_model.eval()
    predictions = []
    with open(args.cls_test_prediction_file, 'w', encoding='utf-8') as output_file:
        with progressbar.ProgressBar(max_value=len(inputs0_list), redirect_stdout=True, redirect_stderr=True) as bar:
            for itr,in0 in enumerate(inputs0_list):
                if (itr % 100) == 0:
                    bar.update(itr)
                with torch.no_grad():
                    output = cls_model.predict(device, in0, input1=(None if (inputs1_list is None) else inputs1_list[itr]))
                if cls_model.token_tagger:
                    output_file.write((' '.join(output)) + '\n')
                    predictions.append(output)
                else:
                    output_file.write(f'{output}' + '\n')
                    predictions.append(output)
    if (args.cls_test_label is not None) and (len(args.cls_test_label)>1):
        ground_truth = [(line.strip().split() if cls_model.token_tagger else (float(line.strip()) if cls_model.is_regression else line.strip())) for line in read_lines(args.cls_test_label)]
        if cls_model.token_tagger:
            F1 = tag_f1_score(ground_truth, predictions, average='weighted', mode='strict')
            print(f'{args.model_variant} {args.model_keyword} {args.task_keyword} TAGGING EVAL F1(weighted) = {(100.0*F1):.2f}\n')
        elif cls_model.is_regression:
            SpearmanR = spearman_corr(np.array(ground_truth), np.array(predictions))
            print(f'{args.model_variant} {args.model_keyword} {args.task_keyword} REGRESSION EVAL REPORT Spearman R = {(100.0*SpearmanR):.2f}')
        else:
            F1 = cls_f1_score(ground_truth, predictions, average='weighted')
            print(f'{args.model_variant} {args.model_keyword} {args.task_keyword} CLASSIFICATION EVAL F1(weighted) = {(100.0*F1):.2f}\n')

