
# Standard libraries
import re

# Non-standard libraries
from sklearn import metrics as skmetrics


################################################################################
#                               Helper Functions                               #
################################################################################
def extract_valid_choice(text, choices=(1, 2, 3)):
    """
    Extract answer from valid choices, as long as only 1 is chosen

    Parameters
    ----------
    text : str
        Arbitrary string
    choices : tuple, optional
        List of options/choices, by default (1, 2, 3)

    Returns
    -------
    str
        Chosen option, if contained. Otherwise returns None
    """
    num_choices_chosen = sum(text.count(str(choice)) > 0 for choice in choices)
    if num_choices_chosen != 1:
        return None
    for choice in choices:
        if str(choice) in text:
            return choice
    raise RuntimeError("Should never reach here!")


def extract_number_from_0_to_99(text):
    """
    Extract number from 0 to 99 from text, as long as there's only 1

    Parameters
    ----------
    text : str
        Arbitrary text

    Returns
    -------
    int
        Extracted number, or None
    """
    pattern = r'\b([0-9]|[1-9][0-9])\b'
    matches = re.findall(pattern, text)
    if len(matches) == 1:
        return matches[0]
    return None


def compute_f1_score_on_zipped_data(label_and_pred):
    """
    Compute F1-score from zipped list of labels and predictions

    Parameters
    ----------
    label_and_pred : tuple
        Contains label and prediction as a tuple

    Returns
    -------
    float
        Micro F1-score
    """
    label, pred = label_and_pred
    return skmetrics.f1_score(label, pred, average="micro")
