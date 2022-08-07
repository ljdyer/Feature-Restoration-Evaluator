# Messages
MESSAGE_CALCULATING_ALL_WERS = """Calculating word error rates for all \
documents..."""
MESSAGE_GETTING_ALL_CMS = "Getting confusion matrices for all documents..."
MESSAGE_INIT_COMPLETE = "Initialisation complete."

# Warnings
WARNING_DIFFERENT_CHARS = """Different characters found between reference and \
hypothesis strings in document index: {doc_idx}! \
(Reference: "{ref_str}"; Hypothesis: "{hyp_str}"). \
Skipping this document (returning None)."""
WARNING_NO_JIWER = """Could not import jiwer library. You will not be able to show \
word error rate info."""

# Errors
ERROR_REF_OR_HYP_TYPE = """
reference and hypothesis parameters must have type list, str, \
or pandas.Series"""
ERROR_NON_EQUAL_LENGTH = \
    "Hypothesis and reference lists must have equal length."
ERROR_CHARS_PER_ROW_AND_NUM_ROWS = \
    "Either none or both of chars_per_row and num_rows must be specified."
