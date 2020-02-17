import pytest

import numpy as np
from correct import Correct

TO_CORRECT = ['DBU (1.5 equlv)', ' NaOH (2 uquvalents)', ' AcOH (5 mo%)', 'BzOH (3 mol%)', ' Ag2O (2.6 equiv.)']


def test_correct_text():
    print(list(Correct(sentence).correct_text() for sentence in TO_CORRECT))
    corrected = [Correct(sentence).correct_text() for sentence in TO_CORRECT]
    print(corrected)
    assert corrected == ['DBU 1.5 equiv', ' NaOH 2 equivalents', ' AcOH 5 mol', 'BzOH 3 mol%',
                         ' Ag2O 2.6 equiv']


if __name__ == '__main__':
    test_correct_text()

