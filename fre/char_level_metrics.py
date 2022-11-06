from typing import Dict, List, Tuple, Union

import jinja2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from fre.misc import CAPS, display_or_print, list_gclust

environment = jinja2.Environment()
template_latex = environment.from_string("""
{% raw %}\hline
& \head{Precision} & \head{Recall} & \head{F-score}
\hline{% endraw %}
{% for feature, scores in prfs.items() -%}
{{feature}} & {{ "%.2f"|format(scores['Precision']) }} & \
{{ "%.2f"|format(scores['Recall']) }} & \
{{ "%.2f"|format(scores['F-score']) }}
{% endfor %}
""")  # noqa: W605

FEATURE_DISPLAY_NAMES = {
    'CAPS': "Capitalization",
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
WARNING_DIFFERENT_CHARS = """
'WARNING: The below characters appear in either the reference or \
hypothesis string but not in both in doc with index {}: {}. \
Returning None.
"""

# === CONFUSION MATRICES ===


# ========================
def show_cms(cms: dict,
             features_to_show: List[str] = None):
    """Show confusion matrices for evaluation results.

    Args:
      cms (dict):
        The confusion matrices.
      features_to_show (List[str]):
        Features to show confusion matrices for. If None, show
        confusion matrics for all features. Defaults to None.
    """

    if features_to_show is None:
        features_to_show = cms.keys()
    for feature in features_to_show:
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
def get_cms(ref: str,
            hyp: str,
            features: list,
            doc_idx: int) -> Dict[str, np.ndarray]:
    """Get confusion matrices for reference and hypothesis strings.

    Args:
      ref (str):
        Reference string
      hyp (str):
        Hypothesis string
      features (list):
        List of features
      doc_idx (int):
        The index of the document (used only for warning messages)

    Returns:
      Dict[str, np.ndarray]:
        The confusion matrices for the document
    """

    chars_ref, feature_lists_ref = get_chars_and_feature_lists(ref, features)
    chars_hyp, feature_lists_hyp = get_chars_and_feature_lists(hyp, features)
    if chars_ref != chars_hyp:
        different_chars = set(chars_ref).symmetric_difference(set(chars_hyp))
        print(WARNING_DIFFERENT_CHARS.format(doc_idx, different_chars))
        return None
    confusion_matrices = {
        f: confusion_matrix(
            [f in x for x in feature_lists_ref],
            [f in x for x in feature_lists_hyp],
            labels=[True, False]
        )
        for f in features
    }
    confusion_matrices['all'] = sum(confusion_matrices[f] for f in features)
    return confusion_matrices


# ====================
def get_chars_and_feature_lists(doc: str,
                                features: List[str]) \
                                    -> Tuple[List[str], List[List]]:

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


# === PRECISON, RECALL, AND F-SCORE ===


# ====================
def show_prfs(cms: Dict[str, np.ndarray],
              for_latex: bool = False):
    """Display a table showing precision, recall, and F-score table.

    Args:
      cms (Dict[str, np.ndarray]):
        Confusion matrices.
      for_latex (bool, optional):
        Whether or not to render the output for LaTeX. Defaults to False.
    """

    prfs = prfs_all_features(cms, display_names=True, for_latex=for_latex)
    if for_latex is True:
        print(template_latex.render(prfs=prfs))
    else:
        display_or_print(pd.DataFrame(prfs).transpose())


# ====================
def prfs_all_features(cms: Dict[str, np.ndarray],
                      display_names: bool = False,
                      for_latex: bool = False) -> Dict[str, dict]:
    """Get precision, recall, and F-score for all features from multiple
    confusion matrics.

    Args:
      cms (Dict[str, np.ndarray]):
        The confusion matrics.
      display_names (bool, optional):
        Whether or not to use display names. Defaults to False.
      for_latex (bool, optional):
        Whether or not to use display names for LaTeX. Defaults to False.

    Returns:
      Dict[str, dict]:
        The precision, recall, and F-score for each feature
    """

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
def prf_single_feature(cm: np.ndarray) -> Dict[str, Union[float, str]]:
    """Calculate precision, recall, and F-score from a confusion matrix.

    Args:
      cm (np.ndarray):
        A confusion matrix

    Returns:
      Dict[str, Union[float, str]]:
        Precision, recall, and F-score.
    """

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
def feature_display_name(feature, latex: bool = False) -> str:
    """Get the display name for a feature (e.g. "Spaces (' ')" for " ")

    Args:
      feature (_type_):
        A single character, or 'CAPS'
      latex (bool, optional):
        Whether to use the feature display name for LaTeX. Defaults to False.

    Returns:
      str:
        The feature display name
    """

    if latex is False and feature in FEATURE_DISPLAY_NAMES:
        return FEATURE_DISPLAY_NAMES[feature]
    elif latex is True and feature in FEATURE_DISPLAY_NAMES_LATEX:
        return FEATURE_DISPLAY_NAMES_LATEX[feature]
    else:
        return f"'{feature}'"
