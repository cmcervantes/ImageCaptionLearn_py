from collections import defaultdict
from Score import Score

"""
Wrapper for easily keeping p/r/f1 scores for
arbitrary labels

@author ccervantes
"""
class ScoreDict:
    def __init__(self, type=int):
        self._goldDict = defaultdict(type)
        self._predDict = defaultdict(type)
        self._correctDict = defaultdict(type)
        self.keys = set()
        self._confusion = defaultdict(type)
    #enddef

    """
    Increments the internal dictionaries with
    a new instance (with gold and predicted label)
    """
    def increment(self, gold_label, pred_label):
        self._goldDict[gold_label] += 1
        self._predDict[pred_label] += 1
        if gold_label == pred_label:
            self._correctDict[gold_label] += 1
        self.keys.add(gold_label)
        self.keys.add(pred_label)
        self._confusion[(gold_label, pred_label)] += 1
    #enddef

    """
    Returns a Score object for a given label
    """
    def getScore(self, label):
        return Score(predicted_count=self._predDict[label],
                     gold_count=self._goldDict[label],
                     correct_count=self._correctDict[label])
    #enddef

    """
    Returns the gold count for the given label, or
    the total number of gold examples, if no label is
    specified
    """
    def getGoldCount(self, label=None):
        if label is None:
            return sum(self._goldDict.values())
        else:
            return self._goldDict[label]
    #enddef

    """
    Returns the gold percentage for a given label, that is
    the percentage of gold examples with this label, over all
    gold examples
    """
    def getGoldPercent(self, label):
        return 100.0 * self.getGoldCount(label) / self.getGoldCount()
    #enddef

    """
    Returns the predicted count for the given label, or
    the total number of predicted examples, if no label is
    specified
    """
    def getPredCount(self, label=None):
        if label is None:
            return sum(self._predDict.values())
        else:
            return self._predDict[label]
    #enddef

    """
    Returns the predicted percentage for a given label, that is
    the percentage of predicted examples with this label, over all
    predicted examples
    """
    def getPredPercent(self, label):
        return 100.0 * self.getPredCount(label) / self.getPredCount()
    #enddef

    """
    Returns the total accuracy for the given ScoreDict
    """
    def getAccuracy(self):
        if sum(self._correctDict.values()) == 0:
            return 0.0
        return 100.0 * sum(self._correctDict.values()) / sum(self._goldDict.values())
    #enddef

    """
    Returns the number of correctly classified items
    """
    def getCorrectCount(self):
        return sum(self._correctDict.values())
    #enddef

    """
    Prints the confusion matrix, where columns are gold, rows are pred
    """
    def printConfusion(self):
        #get the size of the largest count, for formatting
        max_len = 0
        for k in self._confusion.keys():
            k_len = len(str(int(self._confusion[k])))
            if k_len > max_len:
                max_len = k_len
            #endif
        #endfor
        max_len += 9 # to include " (xx.x%) "

        #store the keys as a list, for later
        keys = list(self.keys)

        #Column headers are formatted as
        #      | gold  gold  ... gold
        #and each row will be formatted as
        # pred | count count ... count
        format_str = "%-2s | "
        for i in range(len(keys)):
            format_str += "%-" + str(max_len) + "s"

        # each cell includes a percentage, reflecting
        # how much of the column is accounted for in the cell
        col_totals = defaultdict(int)
        for i in range(len(keys)):
            pred_label = keys[i]
            for j in range(len(keys)):
                gold_label = keys[j]
                col_totals[gold_label] += self._confusion[(gold_label, pred_label)]
            #endfor
        #endfor

        matrix = list()
        col_headers = list()
        col_headers.append("")
        for i in range(len(keys)):
            pred_label = keys[i]

            #since both pred and gold labels are in order, we're
            #adding column headers as we go through the rows
            col_headers.append(str(int(pred_label)))

            #we also want to add this label to the start of the row
            row = list()
            row.append(str(int(pred_label)))
            for j in range(len(keys)):
                gold_label = keys[j]
                count = self._confusion[(gold_label, pred_label)]
                count_str = "%d (%.1f%%)" % (int(count), 100.0 * count / col_totals[gold_label])
                row.append(count_str)
            #endfor
            matrix.append(row)
        #endfor
        matrix.insert(0, col_headers)

        for row in matrix:
            print format_str % tuple(row)
#endclass