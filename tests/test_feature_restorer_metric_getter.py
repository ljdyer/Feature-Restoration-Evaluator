from fre import FeatureRestorationEvaluator     # noqa: E402
from fre.misc import CAPS    # noqa: E402

reference = [
    'This is a sentence.',
    'This is another sentence.',
    'This is Sentence 3'
]
hypothesis = [
    'This is a sentence...',
    'This IS another sentence.',
    'Thisis Senten ce 3'
]
prc = FeatureRestorationEvaluator(
    reference, hypothesis, capitalization=True, feature_chars='., ',
    get_wer_info_on_init=False
)
prfs = prc.get_prfs(0)


# ====================
def test_na():
    """Test that get_prfs returns 'N/A' for punctuation that does
    not appear in the reference sentence"""

    sent_0_commas = prc.get_prfs(0)[',']
    # No commas in reference sentence, so all scores should be N/A
    assert sent_0_commas['Precision'] == 'N/A'
    assert sent_0_commas['Recall'] == 'N/A'
    assert sent_0_commas['F-score'] == 'N/A'


# ====================
def test_ignore_repeated_punctuation():
    """Test that multiple repetitions of the same punctuation mark
    (e.g. '...') are ignored"""

    sent_0_periods = prc.get_prfs(0)['.']
    # Extra periods should be ignored, so P, R, F are all 1.
    assert sent_0_periods['Precision'] == 1
    assert sent_0_periods['Recall'] == 1
    assert sent_0_periods['F-score'] == 1


# ====================
def test_correct_prfs_one():
    """First test that correct precision, recall, and F-scores are
    returned"""

    sent_1_capitalization = prc.get_prfs(1)[CAPS]
    # 1 true positive, 2 false positives, 0 false negatives
    # Precision is tp/(tp+fp) = 1/(2+1) = 1/3
    # Recall is tp/(tp+fn) = 1/(1+0) = 1
    # F-score is (2*p*r)/(p+r) = (2/3)/(4/3) = 1/2
    assert round(sent_1_capitalization['Precision'], 2) == 0.33
    assert sent_1_capitalization['Recall'] == 1
    assert sent_1_capitalization['F-score'] == 0.5


# ====================
def test_correct_prfs_two():
    """Second test that correct precision, recall, and F-scores are
    returned"""

    sent_2_spaces = prc.get_prfs(2)[' ']
    # 2 true positives, 1 false positive, 1 false negative
    # Precision is tp/(tp+fp) = 2/(2+1) = 2/3
    # Recall is tp/(tp+fn) = 2/(2+1) = 2/3
    # F-score is (2*p*r)/(p+r) = (4/9)/(2/3) = 2/3
    assert round(sent_2_spaces['Precision'], 2) == 0.67
    assert round(sent_2_spaces['Recall'], 2) == 0.67
    assert round(sent_2_spaces['F-score'], 2) == 0.67
