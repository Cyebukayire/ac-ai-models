import random
from typing import List, Tuple, Union

from deepkin.clib.libkinlp.kinlpy import ParsedFlexSentence
from deepkin.data.morpho_data import prepare_morpho_data_from_sentence, MorphoDataItem
from deepkin.modules.flex_modules import FlexConfig
from deepkin.utils.arguments import FlexArguments
from deepkin.utils.misc_functions import read_lines, time_now
from torch.utils.data import Dataset, DataLoader


class ClsDataset(Dataset):

    def __init__(self,
                 lines_input0: List[str], lines_input1: List[str] = None,
                 label_dict=None, label_lines=None,
                 regression_target=False,
                 regression_scale_factor=1.0):
        device = 'cpu'
        cfg = FlexConfig()
        assert len(lines_input0) == len(label_lines), "Lines&labels not equal"
        if (lines_input1 is not None):
            assert len(lines_input0) == len(lines_input1), "Lines 0 and 1 not equal"
            self.itemized_data: List[Tuple[Union[int,float],MorphoDataItem]] = [((float(label) / regression_scale_factor) if regression_target else label_dict[label],
                 prepare_morpho_data_from_sentence(cfg, device, ParsedFlexSentence(line0)).add_bos_and_eos().extend(
                     prepare_morpho_data_from_sentence(cfg, device, ParsedFlexSentence(line1)) )) for
                                  label, line0, line1 in zip(label_lines, lines_input0, lines_input1)]
        else:
            self.itemized_data: List[Tuple[Union[int,float],MorphoDataItem]] = [((float(label) / regression_scale_factor) if regression_target else label_dict[label],
                                   prepare_morpho_data_from_sentence(cfg, device, ParsedFlexSentence(line0)).add_bos()) for
                                  label, line0 in zip(label_lines, lines_input0)]
        random.shuffle(self.itemized_data)
        print('Example label:', self.itemized_data[0][0])
        print('Label dict:', label_dict)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

class TagDataset(Dataset):

    def __init__(self,
                 input_lines: List[str],
                 tag_lines: List[str],
                 tags_dict
                 ):
        device = 'cpu'
        self.itemized_data: List[Tuple[List[int],MorphoDataItem]] = []
        cfg = FlexConfig()
        num_lines = len(input_lines)
        assert len(input_lines) == len(tag_lines), "Mismatch between data and labels"
        for idx in range(num_lines):
            if (len(input_lines[idx])>0):
                sentence0 = ParsedFlexSentence(input_lines[idx])
                tags = [tg for tg in tag_lines[idx].split()]
                assert (len(tags) == len(sentence0.tokens)), "Tag misalignment at example # {} '{}'".format(idx+1, input_lines[idx])
                extended_tags_idx = []
                for tag, token in zip(tags, sentence0.tokens):
                    if tag[0] == 'B':
                        extended_tags_idx.append(tags_dict[tag])
                        extended_tags_idx.extend([tags_dict[('I'+tag[1:])]] * (len(token.id_extra_tokens)))
                    else:
                        extended_tags_idx.extend([tags_dict[tag]] * (len(token.id_extra_tokens) + 1))
                assert len(extended_tags_idx) == sum([(len(t.id_extra_tokens)+1) for t in sentence0.tokens]), "Mismatch tags len vs tokens len"
                self.itemized_data.append((extended_tags_idx, prepare_morpho_data_from_sentence(cfg, device, sentence0).add_bos()))
        random.shuffle(self.itemized_data)
        print('Example label:', self.itemized_data[0][0])
        print('Label dict:', tags_dict)

    def __len__(self):
        return len(self.itemized_data)

    def __getitem__(self, idx):
        return self.itemized_data[idx]

def cls_data_collate_wrapper(batch_items: List[Tuple[Union[int,float],MorphoDataItem]]) -> Tuple:
    data = MorphoDataItem('cpu')
    lengths = []
    targets = []
    for label,item in batch_items:
        targets.append(label)
        lengths.append(len(item))
        data.extend(item)
    return data.to_classification_tuple(lengths, targets)

def tag_data_collate_wrapper(batch_items: List[Tuple[List[int],MorphoDataItem]]) -> Tuple:
    data = MorphoDataItem('cpu')
    lengths = []
    targets = []
    for label,item in batch_items:
        targets.extend(label)
        lengths.append(len(item))
        data.extend(item)
    return data.to_classification_tuple(lengths, targets)

def create_cls_data(args: FlexArguments, validation=False):
    labels = args.cls_labels.split(',')
    label_dict = {v.strip(): k for k, v in enumerate(labels)}
    input0 = (args.cls_dev_input0 if validation else args.cls_train_input0)
    input1 = (args.cls_dev_input1 if validation else args.cls_train_input1)
    label = (args.cls_dev_label if validation else args.cls_train_label)
    lines_input0 = read_lines(input0)
    lines_input1 = None if ((input1 is None) or (len(input1)==0)) else read_lines(input1)
    label_lines = read_lines(label)
    dataset = ClsDataset(lines_input0, lines_input1=lines_input1,
                         label_dict=label_dict, label_lines=label_lines,
                         regression_target=args.regression_target,
                         regression_scale_factor=args.regression_scale_factor)
    data_loader = DataLoader(dataset, batch_size=(1 if validation else args.batch_size), collate_fn=cls_data_collate_wrapper,
                             drop_last=(not validation), shuffle=(not validation), pin_memory=args.dataloader_pin_memory,
                             num_workers=(1 if validation else args.dataloader_num_workers),
                             persistent_workers=args.dataloader_persistent_workers)
    if not validation:
        print(time_now(), args.model_variant, f'{len(data_loader)} data loader batches loaded from {label}')
    return dataset, data_loader

def create_tag_data(args: FlexArguments, validation=False):
    labels = args.cls_labels.split(',')
    tags_dict = {v.strip(): k for k, v in enumerate(labels)}
    input0 = (args.cls_dev_input0 if validation else args.cls_train_input0)
    label = (args.cls_dev_label if validation else args.cls_train_label)
    input_lines = read_lines(input0)
    tag_lines = read_lines(label)
    dataset = TagDataset(input_lines, tag_lines, tags_dict)
    data_loader = DataLoader(dataset, batch_size=(1 if validation else args.batch_size), collate_fn=tag_data_collate_wrapper,
                             drop_last=(not validation), shuffle=(not validation), pin_memory=args.dataloader_pin_memory,
                             num_workers=(1 if validation else args.dataloader_num_workers),
                             persistent_workers=args.dataloader_persistent_workers)
    if not validation:
        print(time_now(), args.model_variant, f'{len(data_loader)} data loader batches loaded from {label}')
    return dataset, data_loader