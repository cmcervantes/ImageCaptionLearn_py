
"""
Wrapper for p/r/f1 scores

@author ccervantes
"""
class Score:
    """
    Initializes the Score object with the specified precision and
    recall or computes these values with provided predicted, true,
    and correct counts
    """
    def __init__(self, precision=0.0, recall=0.0,
                 predicted_count=0, gold_count=0,
                 correct_count=0):
        self.p = precision
        self.r = recall
        if predicted_count > 0:
            self.p = float(correct_count) / float(predicted_count)
        if gold_count > 0:
            self.r = float(correct_count) / float(gold_count)
        self.f1 = 0.0
        if self.r > 0 and self.p > 0:
            self.f1 = (2 * self.p * self.r) / (self.p + self.r)
    #enddef

    """
    Returns a score string in the format
    P: PPP.PP% | R: RRR.RR | F1: FFF.FF%
    """
    def toString(self):
        return "P: %6.2f%% | R: %6.2f%% | F1: %6.2f%%" % \
               (100.0 * self.p, 100.0 * self.r, 100.0 * self.f1)
    #enddef

    """
    Returns a score string in the format
    PPP.PP\% & RRR.RR\% & FFF.FF\% \\
    """
    def toLatexString(self):
        return "%6.2f\\%% & %6.2f\\%% & %6.2f\\%% \\\\" % \
               (100.0 * self.p, 100.0 * self.r, 100.0 * self.f1)
    #enddef
#endclass
