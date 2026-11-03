"""Detection precision, recall, and F1.

These are the supervision metrics shipped with the installed package.
Earlier overrides called ``_match_detection_batch``, which supervision 0.31
no longer provides. The public classes already match boxes, respect class
ids, and expose scores at IoU 0.5 and 0.75.
"""

from supervision.metrics import F1Score, Precision, Recall


class MyPrecision(Precision):
    """Precision from the installed supervision version."""


class MyRecall(Recall):
    """Recall from the installed supervision version."""


class MyF1Score(F1Score):
    """F1 from the installed supervision version."""
