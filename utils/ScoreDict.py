from collections import defaultdict
from Score import Score

___author___='ccervantes'


class ScoreDict:
    """
    Wrapper for easily keeping p/r/f1 scores for
    arbitrary labels
    """

    def __init__(self, gold_labels=None, pred_labels=None):
        """
        Initialized the score dict, either empty or populated
        with the optional gold and predicted labels
        :param gold_labels:
        :param pred_labels:
        """
        self._gold_counts = defaultdict(int)
        self._pred_counts = defaultdict(int)
        self._correct_counts = defaultdict(int)
        self.keys = set()
        self._confusion = defaultdict(int)

        if gold_labels is not None and pred_labels is not None:
            for idx in range(0, len(gold_labels)):
                self.increment(gold_labels[idx], pred_labels[idx])
    #enddef

    def increment(self, gold_label, pred_label):
        """
        Increments the internal dictionaries with
        a new instance (with gold and predicted label)

        :param gold_label:
        :param pred_label:
        :return:
        """
        self._gold_counts[gold_label] += 1
        self._pred_counts[pred_label] += 1
        if gold_label == pred_label:
            self._correct_counts[gold_label] += 1
        self.keys.add(gold_label)
        self.keys.add(pred_label)
        self._confusion[(gold_label, pred_label)] += 1
    #enddef

    def merge(self, score_dict):
        """
        Increments the internal dictionaries with
        all the instances in the given score_dict

        :param score_dict:
        :return:
        """
        for label in score_dict._gold_counts.keys():
            self._gold_counts[label] += score_dict._gold_counts[label]
            self.keys.add(label)
        for label in score_dict._pred_counts.keys():
            self._pred_counts[label] += score_dict._pred_counts[label]
            self.keys.add(label)
        for label in score_dict._correct_counts.keys():
            self._correct_counts[label] += score_dict._correct_counts[label]
            self.keys.add(label)
        for labels in score_dict._confusion.keys():
            self._confusion[labels] += score_dict._confusion[labels]
    #enddef

    def get_score(self, label):
        """
        Returns a Score object for a given label

        :param label:
        :return:
        """
        return Score(predicted_count=self._pred_counts[label],
                     gold_count=self._gold_counts[label],
                     correct_count=self._correct_counts[label])
    #enddef

    def get_gold_count(self, label=None):
        """
        Returns the gold count for the given label, or
        the total number of gold examples, if no label is
        specified

        :param label:
        :return:
        """
        if label is None:
            return sum(self._gold_counts.values())
        else:
            return self._gold_counts[label]
    #enddef

    def get_gold_percent(self, label):
        """
        Returns the gold percentage for a given label, that is
        the percentage of gold examples with this label, over all
        gold examples

        :param label:
        :return:
        """
        return 100.0 * self.get_gold_count(label) / self.get_gold_count()
    #enddef

    def get_pred_count(self, label=None):
        """
        Returns the predicted count for the given label, or
        the total number of predicted examples, if no label is
        specified

        :param label:
        :return:
        """
        if label is None:
            return sum(self._pred_counts.values())
        else:
            return self._pred_counts[label]
    #enddef

    def get_pred_percent(self, label):
        """
        Returns the predicted percentage for a given label, that is
        the percentage of predicted examples with this label, over all
        predicted examples

        :param label:
        :return:
        """
        return 100.0 * self.get_pred_count(label) / self.get_pred_count()
    #enddef

    def get_accuracy(self):
        """
        Returns the total accuracy for the given ScoreDict

        :return:
        """
        if sum(self._correct_counts.values()) == 0:
            return 0.0
        return 100.0 * sum(self._correct_counts.values()) / sum(self._gold_counts.values())
    #enddef

    def get_correct_count(self):
        """
        Returns the number of correctly classified items

        :return:
        """
        return sum(self._correct_counts.values())
    #enddef

    def print_confusion(self):
        """
        Prints the confusion matrix, where columns are gold, rows are pred

        :return:
        """
        # get the size of the largest count, for formatting
        max_len = 0
        for k in self._confusion.keys():
            k_len = len(str(int(self._confusion[k])))
            if k_len > max_len:
                max_len = k_len
            #endif
        #endfor
        max_len += 9 # to include " (xx.x%) "

        # store the keys as a list, for later
        keys = list(self.keys)

        # Confusion matrix formatting is
        #        | gold  gold  ... gold
        #   pred | count count ... count
        format_str = "%-" + str(max_len) + "s | "
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

            # since both pred and gold labels are in order, we're
            # adding column headers as we go through the rows
            col_headers.append(str(pred_label))

            # we also want to add this label to the start of the row
            row = list()
            row.append(str(pred_label))
            for j in range(len(keys)):
                gold_label = keys[j]
                count = self._confusion[(gold_label, pred_label)]
                perc = 0.0
                if col_totals[gold_label] > 0:
                    perc = 100.0 * count / col_totals[gold_label]
                count_str = "%d (%.1f%%)" % (int(count), perc)
                row.append(count_str)
            #endfor
            matrix.append(row)
        #endfor
        matrix.insert(0, col_headers)

        for row in matrix:
            print format_str % tuple(row)
    #enddef
#endclass