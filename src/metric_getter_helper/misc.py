import pickle
from typing import Any, Union

import pandas as pd
from tqdm import tqdm as non_notebook_tqdm
from tqdm.notebook import tqdm as notebook_tqdm

try:
    from IPython.core.display import HTML
except:
    pass

from metric_getter_helper.messages import (ERROR_REF_OR_HYP_TYPE,
                                           WARNING_DIFFERENT_CHARS)

Int_or_Str = Union[int, str]
Str_or_List = Union[str, list]
Str_or_List_or_Series = Union[str, list, pd.Series]


# ====================
def check_same_char(next_char: dict, chars: dict,
                    doc_idx: str = 'UNKNOWN') -> bool:

    try:
        assert next_char['ref'].lower() == next_char['hyp'].lower()
        return True
    except AssertionError:
        error_msg = WARNING_DIFFERENT_CHARS.format(
            doc_idx=doc_idx,
            ref_str=(next_char['ref'] + ''.join(chars['ref'][:10])),
            hyp_str=(next_char['hyp'] + ''.join(chars['hyp'][:10]))
        )
        print(error_msg)
        return False


# ====================
def str_or_list_or_series_to_list(
     input_: Str_or_List_or_Series) -> list:

    if isinstance(input_, str):
        return [input_]
    elif isinstance(input_, pd.Series):
        return input_.to_list()
    elif isinstance(input_, list):
        return input_
    else:
        raise TypeError(ERROR_REF_OR_HYP_TYPE)


# ====================
def get_tqdm() -> type:
    """Return tqdm.notebook.tqdm if code is being run from a notebook,
    or tqdm.tqdm otherwise"""

    if is_running_from_ipython():
        tqdm_ = notebook_tqdm
    else:
        tqdm_ = non_notebook_tqdm
    return tqdm_


# ====================
def is_running_from_ipython():
    """Determine whether or not the current script is being run from
    a notebook"""

    try:
        # Notebooks have IPython module installed
        from IPython import get_ipython
        return True
    except ModuleNotFoundError:
        return False


# ====================
def display_or_print(obj: Any):
    """'print' or 'display' an object, depending on whether the current
    script is running from a notebook or not."""

    if is_running_from_ipython():
        display(obj)
    else:
        print(obj)


# ====================
def display_or_print_html(html: str):
    """'print' or 'display' an object, depending on whether the current
    script is running from a notebook or not."""

    if is_running_from_ipython():
        display(HTML(html))
    else:
        print(html)


# ====================
def load_pickle(fp: str) -> Any:
    """Load a .pickle file and return the data"""

    with open(fp, 'rb') as f:
        unpickled = pickle.load(f)
    return unpickled


# ====================
def save_pickle(data: Any, fp: str):
    """Save data to a .pickle file"""

    with open(fp, 'wb') as f:
        pickle.dump(data, f)
