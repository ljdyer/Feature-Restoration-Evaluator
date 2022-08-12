# Feature Restorer Metric Getter

A Python class for calculating precision/recall/F-score and word error rate (WER) metrics for the outputs of feature restoration models against reference strings.

## Getting started

### 1. Clone the repository

Recommended method for Google Colab notebooks:

```python
import sys
# Delete feature-restorer-metric-getter folder to ensures that any changes to the repo are reflected
!rm -rf 'feature-restorer-metric-getter'
# Clone feature-restorer-metric-getter repo
!git clone https://github.com/ljdyer/feature-restorer-metric-getter.git
# Add feature-restorer-metric-getter to PYTHONPATH
sys.path.append('feature-restorer-metric-getter/src')
```

### 2. Install requirements (if required)

If working in Google Colab, the only requirement is `jiwer`, as all other dependencies are installed by default

```python
!pip install jiwer
```

`jiwer` is only required for WER-related features, so you can still use Feature Restorer Metric Getter to calculate precision/recall/F-score metrics without it.

If working in a virtual environment, run the following in the src directory:

```python
pip install -r requirements.txt
```

### 3. Import FeatureRestorerMetricGetter class

```python
from feature_restorer_metric_getter import FeatureRestorerMetricGetter
```

## How to use

### Initializing a class instance

```python
# ====================
class PrecisionRecallCalculator:

    # ====================
    def __init__(self,
                 reference: Str_or_List_or_Series,
                 hypothesis: Str_or_List_or_Series,
                 capitalisation: bool,
                 feature_chars: Str_or_List,
                 get_cms_on_init: bool = True):
        """
        Initialize an instance of the PrecisionRecallCalculator class

        Required arguments:
        -------------------
        reference:                  Either a single string, or a list or
            Str_or_List_or_Series   pandas.Series object of strings
                                    ('documents') to use as the reference
                                    corpus.
        hypothesis:                 Either a single string, or a list or
            Str_or_List_or_Series   pandas.Series object of strings
                                    ('documents') to use as the hypothesis
                                    corpus.
                                    (Number of documents must be the same
                                    as reference.)
        capitalisation: bool        Whether or not to treat capitalisation
                                    as a feature to be assessed.
        feature_chars:              A string or list of characters containing
            Str_or_List             other characters to treat as features
                                    (e.g. '., ' for periods, commas, and
                                    spaces.)

        Optional keyword arguments:
        ---------------------------
        get_cms_on_init: bool       Whether or not to get confusion matrices
                                    for all reference/hypothesis documents
                                    on intiialization. Set to false and access
                                    manually to save time if only looking at
                                    metrics for a subset of documents in a
                                    large corpus.
        """
```

#### Example usage:

```python
RESULTS_DF_PATH = 'drive/MyDrive/Group Assignment/Results/end_to_end.csv'
results_df_csv = pd.read_csv(RESULTS_DF_PATH)
reference = results_df_csv['reference']
hypothesis = results_df_csv['model_5_result']
prc_TED = PrecisionRecallCalculator(
    reference, hypothesis, True, '., ')
```

<img src="readme-img/init.PNG"></img>

### Displaying precision, recall, and F-score metrics

```python
    # ====================
    def show_precision_recall_fscore(self, doc_idx: Int_or_Str = 'all'):
        """
        Show precision, recall and F-score for each feature, for
        either a single document or the entire corpus.

        Optional keyword arguments:
        ---------------------------
        doc_idx: Int_or_Str         Either an integer indicating the index of
                                    the document to show metrics for, or 'all'
                                    to show metrics for all documents in the
                                    corpus (the default behaviour).
        """

        feature_scores = {
            self.feature_display_name(feature):
            self.precision_recall_fscore_from_cm(
                self.confusion_matrices[doc_idx][feature])
            for feature in self.features + ['all']}
        display_or_print(pd.DataFrame(feature_scores).transpose())
```

#### Example usage:

```python
prc_TED.show_precision_recall_fscore()
```

<img src="readme-img/metrics.PNG"></img>

### Displaying confusion matrices

```python
    # ====================
    def show_confusion_matrices(self, doc_idx: Int_or_Str = 'all'):
        """
        Show confusion matrices for each feature, for either a
        single document or the entire corpus.

        Optional keyword arguments:
        ---------------------------
        doc_idx: Int_or_Str         Either an integer indicating the index of
                                    the document to show confusion matrices
                                    for, or 'all' to show confusion matrices
                                    for all documents in the corpus (the
                                    default behaviour).
        """
```

#### Example usage:

```python
prc_TED.show_confusion_matrices()
```

<img src="readme-img/confusion_matrices.PNG"></img>