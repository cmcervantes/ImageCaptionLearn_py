___author___ = 'ccervantes'


class Score:
    def __init__(self, precision=0.0, recall=0.0,
                 predicted_count=0, gold_count=0,
                 correct_count=0):
        """
        Initializes the Score object with the specified precision and
        recall or computes these values with provided predicted, true,
        and correct counts
        :param precision:
        :param recall:
        :param predicted_count:
        :param gold_count:
        :param correct_count:
        """
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

    def to_string(self):
        """
        Returns a score string in the format
        P: PPP.PP% | R: RRR.RR | F1: FFF.FF%
        :return:
        """
        return "P: %6.2f%% | R: %6.2f%% | F1: %6.2f%%" % \
               (100.0 * self.p, 100.0 * self.r, 100.0 * self.f1)
    #enddef

    def to_latex_string(self):
        """
        Returns a score string in the format
        PPP.PP\% & RRR.RR\% & FFF.FF\% \\
        :return:
        """
        return "%6.2f\\%% & %6.2f\\%% & %6.2f\\%% \\\\" % \
               (100.0 * self.p, 100.0 * self.r, 100.0 * self.f1)
    #enddef
#endclass
