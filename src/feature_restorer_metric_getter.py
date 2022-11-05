"""
feature_restorer_metric_getter.py

Main module for FeatureRestorerMetricGetter class
"""

from src.confusion_matrices import (cms, prfs_all_features, show_cm_tables,
                                     show_prfs)
from src.misc import (CAPS, Int_or_Str, Str_or_List, Str_or_List_or_Series,
                       get_tqdm, load_pickle, save_pickle,
                       str_or_list_or_series_to_list)
from src.text_display import show_feature_errors_, show_text_display_
from src.word_error_rate import show_wer_info_table, wer, wer_info

tqdm_ = get_tqdm()

# Messages
MESSAGE_CALCULATING_ALL_WERS = """Calculating word error rates for all \
documents..."""
MESSAGE_GETTING_ALL_CMS = "Getting confusion matrices for all documents..."
MESSAGE_INIT_COMPLETE = "Initialisation complete."


# ====================
class FeatureRestorerMetricGetter:

    # ====================
    def __init__(self,
                 reference: Str_or_List_or_Series,
                 hypothesis: Str_or_List_or_Series,
                 capitalisation: bool,
                 feature_chars: Str_or_List,
                 get_cms_on_init: bool = True,
                 get_wer_info_on_init: bool = True):
        """Initalises FeatureRestorerMetricGetter.

        Args:
          reference (Str_or_List_or_Series):
            Either a single string, or a list or pandas.Series object of
            strings ('documents') to use as the reference corpus.
          hypothesis (Str_or_List_or_Series):
            Either a single string, or a list or pandas.Series object of
            strings ('documents') to use as the hypothesis corpus.
            (Number of documents must be the same as reference.)
          capitalisation (bool):
            Whether or not to treat capitalisation as a feature to be assessed.
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

        self.reference = str_or_list_or_series_to_list(reference)
        self.hypothesis = str_or_list_or_series_to_list(hypothesis)
        if len(self.reference) != len(self.hypothesis):
            raise ValueError(
                "Hypothesis and reference lists must have equal length."
            )
        self.feature_chars = list(feature_chars)
        self.set_features(capitalisation)
        self.wer_info = {}
        self.cms = {}
        if get_wer_info_on_init:
            self.get_wer_info_all()
        if get_cms_on_init:
            self.get_cms_all()
        print(MESSAGE_INIT_COMPLETE)

    # ====================
    @classmethod
    def from_pickle(cls, load_path: str):
        """Constructor to create a class method from a pickle file.

        Args:
          load_path (str): The path to the pickle file.

        Returns:
          FeatureRestorerMetricGetter: The constructed class instance.
        """

        self = cls.__new__(cls)
        data = load_pickle(load_path)
        self.__dict__.update(data)
        return self

    # ====================
    def to_pickle(self, save_path: str):
        """Save the attributes of the current class instance to a pickle file.

        Args:
          save_path (str):
            The path to save to.
        """

        data = self.__dict__
        save_pickle(data, save_path)

    # ====================
    def set_features(self, capitalisation: bool):
        """Set self.features attribute

        Args:
          capitalisation (bool):
            If True, add CAPS to the list of feature_chars.
        """

        if capitalisation:
            self.features = \
                [CAPS] + self.feature_chars.copy()
        else:
            self.features = self.feature_chars.copy()

    # === WORD ERROR RATE ===

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

        self.get_wer_info(doc_idx)
        wer_info = self.wer_info[doc_idx]
        if for_latex is True:
            print(rf"\textbf{{WER:}} {wer_info['wer']:.2f}\%\\")
        else:
            show_wer_info_table(wer_info)

    # ====================
    def get_wer_info(self, scope: Int_or_Str):

        if scope == 'all':
            self.get_wer_info_all()
        else:
            self.get_wer_info_doc(scope)

    # ====================
    def get_wer_info_all(self):
        """Calculate reference length, minimum number of edits, and word
        error rate for all documents.
        """

        # Get WER info for each document
        print(MESSAGE_CALCULATING_ALL_WERS)
        for doc_idx in tqdm_(range(len(self.hypothesis))):
            if doc_idx not in self.wer_info:
                self.get_wer_info_doc(doc_idx)
        # Get overall WER info
        all_doc_idxs = range(len(self.reference))
        len_ref_all = sum([
            self.wer_info[doc_idx]['len_ref'] for doc_idx in all_doc_idxs
        ])
        num_edits_all = sum([
            self.wer_info[doc_idx]['num_edits'] for doc_idx in all_doc_idxs
        ])
        wer_all = wer(num_edits_all, len_ref_all)
        self.wer_info['all'] = {
            'len_ref': len_ref_all,
            'num_edits': num_edits_all,
            'wer': wer_all
        }

    # ====================
    def get_wer_info_doc(self, doc_idx: int):
        """Calculate reference length, minimum number of edits, and word
        error rate for a single document
        """

        ref = self.reference[doc_idx].strip()
        hyp = self.hypothesis[doc_idx].strip()
        wer_info_ = wer_info(ref, hyp)
        self.wer_info[doc_idx] = wer_info_

    # === CONFUSION MATRICES ===

    # ====================
    def show_confusion_matrices(self, doc_idx: Int_or_Str = 'all'):
        """Show confusion matrices for each feature, for either a
        single document or all documents.

        Args:
          doc_idx (Int_or_Str, optional):
            Either an integer indicating the index of the document to
            show confusion matrices for, or 'all' to show confusion
            matrices for all documents in the corpus. Defaults to 'all'.
        """

        self.get_cms(doc_idx)
        cms = self.cms[doc_idx]
        show_cm_tables(cms)

    # ====================
    def get_cms(self, scope: Int_or_Str):

        if scope == 'all':
            self.get_cms_all()
        else:
            self.get_cms_doc(scope)

    # ====================
    def get_cms_all(self):
        """Get confusion matrices for all documents.
        """

        # Get confusion matrices for each document
        print(MESSAGE_GETTING_ALL_CMS)
        for doc_idx in tqdm_(range(len(self.hypothesis))):
            if doc_idx not in self.cms:
                self.get_cms_doc(doc_idx)
        all_docs = {}
        # Get overall confusion matrices
        for f in self.features + ['all']:
            all_docs[f] = \
                sum([self.cms[doc_idx][f]
                    for doc_idx in range(len(self.reference))
                    if self.cms[doc_idx] is not None])
        self.cms['all'] = all_docs

    # ====================
    def get_cms_doc(self, doc_idx: int) -> dict:
        """Get confusion matrices for a single document.

        Args:
          doc_idx (int): The index of the document to get
          confusion matrics for.

        Returns:
          dict:
            A dictionary containing the confusion matrices.
        """

        cm_doc = cms(
            self.reference[doc_idx].strip(),
            self.hypothesis[doc_idx].strip(),
            self.features,
            doc_idx
        )
        self.cms[doc_idx] = cm_doc
        return cm_doc

    # === PRECISON, RECALL, AND F-SCORE ===

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

        self.get_cms(doc_idx)
        cms = self.cms[doc_idx]
        show_prfs(cms, for_latex)

    # ====================
    def get_prfs(self,
                 doc_idx: Int_or_Str = 'all',
                 display_names: bool = False) -> dict:

        self.get_cms(doc_idx)
        cms = self.cms[doc_idx]
        return prfs_all_features(cms, display_names)

    # === TEXT_DISPLAY ===

    # ====================
    def show_text_display(self,
                          doc_idx: int,
                          start_char: int = None,
                          chars_per_row: int = None,
                          num_rows: int = None,
                          for_latex: bool = False,
                          ignore: list = None):

        ref = self.reference[doc_idx].strip()
        hyp = self.hypothesis[doc_idx].strip()
        show_text_display_(
            ref, hyp,
            features=self.features, feature_chars=self.feature_chars,
            start_char=start_char, chars_per_row=chars_per_row,
            num_rows=num_rows, ignore=ignore, for_latex=for_latex
        )

    # ====================
    def show_feature_errors(self,
                            doc_idx: int,
                            feature_to_check: str,
                            chars_either_side: int = 10):

        ref = self.reference[doc_idx].strip()
        hyp = self.hypothesis[doc_idx].strip()
        show_feature_errors_(
            ref, hyp,
            features=self.features,
            feature_to_check=feature_to_check,
            chars_either_side=chars_either_side
        )
