from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from metric_getter_helper.misc import (CAPS, display_or_print,
                                       list_gclust)

import jinja2

environment = jinja2.Environment()
template_latex = environment.from_string("""
\hline
& \head\{Precision\} & \head\{Recall\} & \head\{F-score\}
\hline
{% for feature, scores in prfs %}
{{feature}}
{% endfor %}
""")

#  & {{scores['Precision']:.2f}} & {{scores['Recall']:.2f}} & {{scores['F-score']:.2f}}

FEATURE_DISPLAY_NAMES = {
    'CAPS': "Capitalisation",
    ' ': "Spaces (' ')",
    ',': "Commas (',')",
    '.': "Periods ('.')",
    'all': 'All features'
}
FEATURE_DISPLAY_NAMES_LATEX = {
    ' ': r"Spaces ('{\ }')",
    ',': "Commas (',')",
    '.': "Periods ('.')",
    'all': 'All'
}

ERROR_FIRST_CHAR_FEATURE_CHAR = """
The first character in the document is a feature character!"""


# ====================
def cms(ref: str, hyp: str, features: list, doc_idx: int):

    chars_ref, feature_lists_ref = get_chars_and_feature_lists(ref, features)
    chars_hyp, feature_lists_hyp = get_chars_and_feature_lists(hyp, features)
    assert chars_ref == chars_hyp
    confusion_matrices = {
        f: confusion_matrix(
            [f in x for x in feature_lists_ref],
            [f in x for x in feature_lists_hyp],
            labels=[True, False]
        )
        for f in features
    }
    confusion_matrix_all = sum(
        confusion_matrices[f] for f in features)
    confusion_matrices['all'] = confusion_matrix_all
    return confusion_matrices


# ====================
def get_chars_and_feature_lists(doc: str,
                                features: List[str]) -> Tuple[List[str]]:

    chars = list_gclust(doc.strip())
    non_feature_chars = []
    feature_lists = []
    # Check whether first char is a feature char
    if chars[0] in features:
        raise ValueError(ERROR_FIRST_CHAR_FEATURE_CHAR)
    while chars:
        next_char = chars.pop(0)
        non_feature_chars.append(next_char)
        feature_lists.append([])
        if next_char.isupper():
            feature_lists[-1].append(CAPS)
        while (len(chars) > 0 and chars[0] in features):
            feature_lists[-1].append(chars.pop(0))
    return list(map(lambda x: x.lower(), non_feature_chars)), feature_lists


# ========================
def show_cm_tables(cms: dict):

    for feature, cm in cms.items():
        display_name = feature_display_name(feature)
        print(display_name)
        print('=' * len(display_name))
        print()
        cm = cms[feature]
        col_index = pd.MultiIndex.from_tuples(
            [('Hypothesis', 'positive'), ('Hypothesis', 'negative')])
        row_index = pd.MultiIndex.from_tuples(
            [('Reference', 'positive'), ('Reference', 'negative')])
        display_or_print(pd.DataFrame(
            cm, index=row_index, columns=col_index))
        print()


# ====================
def show_prfs(cms, for_latex: bool = False):

    prfs = prfs_all_features(cms, display_names=True, for_latex=for_latex)
    if for_latex is True:
        template_latex.render(prfs=prfs)
    else:
        display_or_print(pd.DataFrame(prfs).transpose())


# ====================
def show_prfs_latex(prfs: dict):

    output_lines = []
    output_lines.append(r"\hline")
    output_lines.append(r"& \head{Precision} & \head{Recall} & " +
                        r"\head{F-score}\\")
    output_lines.append(r"\hline")
    for feature, scores in prfs.items():
        new_line = (rf"{feature} & {scores['Precision']:.2f} & " +
                    rf"{scores['Recall']:.2f} & " +
                    rf"{scores['F-score']:.2f}\\")
        output_lines.append(new_line)
    print('\n'.join(output_lines))


# ====================
def prfs_all_features(cms,
                      display_names: bool = False,
                      for_latex: bool = False):

    if display_names is True:
        prfs = {
            feature_display_name(feature, for_latex):
                prf_single_feature(cm)
            for feature, cm in cms.items()
        }
    else:
        prfs = {
            feature: prf_single_feature(cm)
            for feature, cm in cms.items()
        }
    return prfs


# ====================
def prf_single_feature(cm: np.ndarray):
    """Calculate precision, recall, and F-score from a confusion matrix."""

    tp = float(cm[0][0])
    fp = float(cm[1][0])
    fn = float(cm[0][1])
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 'N/A'
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 'N/A'
    try:
        fscore = (2*precision*recall) / (precision+recall)
    except (TypeError, ZeroDivisionError):
        fscore = 'N/A'
    return {
        'Precision': precision,
        'Recall': recall,
        'F-score': fscore,
    }


# ========================
def feature_display_name(feature, latex: bool = False):
    """Return the display name for a feature."""

    if latex is False and feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]
    elif latex is True and feature in FEATURE_DISPLAY_NAMES_LATEX:
        return FEATURE_DISPLAY_NAMES_LATEX[feature]
    else:
        return f"'{feature}'"
