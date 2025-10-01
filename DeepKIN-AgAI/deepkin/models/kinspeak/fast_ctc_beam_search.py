import math
from dataclasses import dataclass

import torch

from deepkin.data.syllabe_vocab import BLANK_ID, KINSPEAK_VOCAB_IDX


def log(x: float) -> float:
    return math.log(x+1e-32)

def logaddexp(x:float, y:float) -> float:
    return torch.logaddexp(torch.tensor(x), torch.tensor(y)).item()

@dataclass
class BeamEntry:
    """Information about one single beam at specific time-step."""
    pr_total: float = log(0)  # blank and non-blank
    pr_non_blank: float = log(0)  # non-blank
    pr_blank: float = log(0)  # blank
    pr_text: float = log(1)  # LM score
    lm_applied: bool = False  # flag if LM was already applied to this beam
    labeling: tuple = ()  # beam-labeling
    def __lt__(self, other):
        return (self.pr_total + self.pr_text) < (other.pr_total + other.pr_text)



def ctc_beam_search(log_probs_table: torch.Tensor,
                    beam_width: int = 25,
                    blank_idx: int = 6):
    max_T, max_C = log_probs_table.size(0), log_probs_table.size(1)

    # initialise beam state
    entry = BeamEntry()
    entry.pr_blank = log(1)
    entry.pr_total = log(1)
    last_list = [entry]

    # go over all time-steps
    for t in range(max_T):
        current_list = []
        last_list = sorted(last_list, reverse=True)[:beam_width]
        # go over best beams
        for entry in last_list:
            labeling, pr_total = entry.labeling, entry.pr_total

            # probability of paths ending with a non-blank
            pr_non_blank = log(0)
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                pr_non_blank = entry.pr_non_blank + log_probs_table[t, labeling[-1]].item()

            # probability of paths ending with a blank
            pr_blank = entry.pr_total + log_probs_table[t, blank_idx].item()

            # fill in data for current beam
            new_entry = BeamEntry()
            new_entry.labeling = labeling
            new_entry.pr_non_blank = logaddexp(new_entry.pr_non_blank, pr_non_blank)
            new_entry.pr_blank = logaddexp(new_entry.pr_blank, pr_blank)
            new_entry.pr_total = logaddexp(new_entry.pr_total,
                                                           logaddexp(pr_blank, pr_non_blank))
            new_entry.pr_text = entry.pr_text
            new_entry.lm_applied = True  # LM already applied at previous time-step for this beam-labeling

            current_list.append(new_entry)

            # extend current beam-labeling
            breaking = False
            for c,vl in KINSPEAK_VOCAB_IDX.items():
                if c != BLANK_ID:
                    # add new char to current beam-labeling
                    new_labeling = labeling + (c,)

                    # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                    if labeling and labeling[-1] == c:
                        pr_non_blank = entry.pr_blank + log_probs_table[t, c].item()
                    else:
                        pr_non_blank = entry.pr_total + log_probs_table[t, c].item()

                    # fill in data
                    additional_entry = BeamEntry()
                    additional_entry.labeling = new_labeling
                    additional_entry.pr_non_blank = logaddexp(additional_entry.pr_non_blank,
                                                                           pr_non_blank)
                    additional_entry.pr_total = logaddexp(additional_entry.pr_total, pr_non_blank)
                    current_list.append(additional_entry)
        # set new beam state
        last_list = current_list

    # normalise LM scores according to beam-labeling-length
    for entry in last_list:
        labeling_len = len(entry.labeling)
        entry.pr_text = (1.0 / (labeling_len if labeling_len else 1.0)) * entry.pr_text

    # sort by probability
    last_list = sorted(last_list, reverse=True)
    best_labeling = last_list[0].labeling
    return list(best_labeling)
