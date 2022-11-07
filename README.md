# Feature Restoration Evaluator

A Python library for quantitative and qualitative evaluation of restoration of textual features using machine learning models.

Developed and used for the paper "Comparison of Token- and Character-Level Approaches to Restoration of Spaces, Punctuation, and Capitalization in Various Languages", which is scheduled for publication in December 2022.

## Interactive demo

Check out the interactive demo <a href="https://colab.research.google.com/drive/1JkQAEH2uNDQkVl7BNj8vrsOeFbSeGNn_?usp=sharing" target="_blank">here</a> to try out the library for yourself in a Google Colab notebook using sample data from the results of the models discussed in the paper.

Alternatively, scroll down for instructions on getting started and basic documentation.

## Getting started

### Install the library using `pip`

```
!pip install git+https://github.com/ljdyer/feature-restoration-evaluator.git
```

### Import the `FeatureRestorationEvaluator` class

```python
from fre import FeatureRestorationEvaluator
```

## Evaluate feature restorations using the `FeatureRestorationEvaluator` class

### Initialize an instance of the FeatureRestorationEvaluator class

#### FeatureRestorationMetricEvaluator.\_\_init\_\_

```python
    # ====================
    def __init__(self,
                 reference: Str_or_List_or_Series,
                 hypothesis: Str_or_List_or_Series,
                 capitalization: bool,
                 feature_chars: Str_or_List,
                 get_cms_on_init: bool = True,
                 get_wer_info_on_init: bool = True):
        """Initalises FeatureRestorationEvaluator.

        Args:
          reference (Str_or_List_or_Series):
            Either a single string, or a list or pandas.Series object of
            strings ('documents') to use as the reference corpus.
          hypothesis (Str_or_List_or_Series):
            Either a single string, or a list or pandas.Series object of
            strings ('documents') to use as the hypothesis corpus.
            (Number of documents must be the same as reference.)
          capitalization (bool):
            Whether or not to treat capitalization as a feature to be assessed.
          feature_chars (Str_or_List):
            A string or list of characters containing other characters to treat
            as features (e.g. '., ' for periods, commas, and spaces.)
          get_cms_on_init (bool, optional):
            Whether or not to get confusion matrices for all
            reference/hypothesis documents on intiialization. Set to False to
            save time if you do not need precision, recall, and F-score
            information or only need it for a subset of documents. Defaults to
            True.
          get_wer_info_on_init (bool, optional):
            Whether or not to calculate WERs for all reference/hypothesis
            documents on initialization. Set to False to save time if you do
            not need WER information or only need WER information for a subset
            of documents. Defaults to True.

        Raises:
          ValueError:
            Hypothesis and reference lists must have equal length.
        """
```

#### Example usage:

```python
TEST_PATH = 'drive/MyDrive/PAPER/data/ted_talks/ted_test.csv'
RESULTS_PATH = 'drive/MyDrive/PAPER/models/05_bilstm_e2e/english/TedTalks/results.csv'
reference = pd.read_csv(TEST_PATH)['all_cleaned'].to_list()
hypothesis = pd.read_csv(RESULTS_PATH)['results'].to_list()
frmg = FeatureRestorationMetricGetter(reference, hypothesis, True, '., ', False, False)
```

<img src="readme-img/init.PNG"></img>

### Displaying precision, recall, and F-score metrics

#### FeatureRestorationMetricEvaluator.show_prfs

```python
    # ====================
    def show_prfs(self,
                  doc_idx: Int_or_Str = 'all',
                  for_latex: bool = False):
        """Show precision, recall and F-score for each feature, for
        either a single document all documents.

        Args:
          doc_idx (Int_or_Str, optional):
            Either an integer indicating the index of the document to
            show metrics for, or 'all' to show metrics for all documents
            in the corpus. Defaults to 'all'.
          for_latex (bool, optional):
            Whether or not to format the output for LaTeX.
            Defaults to False.
        """
```

#### Example usage:

```python
src.show_prfs()
```

<img src="readme-img/show_prfs.PNG"></img>

### Displaying confusion matrices

#### FeatureRestorationMetricEvaluator.show_confusion_matrices

```python
    # ====================
    def show_confusion_matrices(self,
                                doc_idx: Int_or_Str = 'all',
                                features_to_show: List[str] = None):
        """Show confusion matrices for each feature, for either a
        single document or all documents.

        Args:
          doc_idx (Int_or_Str, optional):
            Either an integer indicating the index of the document to
            show confusion matrices for, or 'all' to show confusion
            matrices for all documents in the corpus. Defaults to 'all'.
          features_to_show (List[str]):
            Features to show confusion matrices for. If None, show
            confusion matrics for all features. Defaults to None.
        """
```

#### Example usage:

```python
fmrg.show_confusion_matrices()
```

### Displaying WER information

#### FeatureRestorationMetricEvaluator.show_wer_info

```python
    # ====================
    def show_wer_info(self,
                      doc_idx: Int_or_Str = 'all',
                      for_latex: bool = False):
        """Show minimum edit distance, reference length, and word error rate
        for either a single document or all documents.

        Args:
          doc_idx (Int_or_Str, optional):
            Either an integer indicating the index of the document to show
            confusion matrices for, or 'all' to show confusion matrices for
            all documents in the corpus.
          for_latex (bool, optional):
            Whether or not to format the output for LaTeX. Defaults to False.
        """
```

#### Example usage:

```python
fmrg.show_wer_info()
```

<img src="readme-img/show_wer.PNG"></img>
