import os
import sys
import numpy as np
from collections import Counter
import random

try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp

# helpers to load data
from data_helper import load_vote_data, load_incomplete_entry
# helpers to learn and traverse the tree over attributes
from tree import get_mst, get_tree_root, get_tree_edges

# pseudocounts for uniform dirichlet prior (Laplace smoothing)
alpha = 0.1


def renormalize(cnt):
    '''
    renormalize a Counter()
    '''
    tot = 1. * sum(cnt.values())
    for a_i in cnt:
        cnt[a_i] /= tot
    return cnt

# --------------------------------------------------------------------------
# Naive bayes CPT and classifier
# --------------------------------------------------------------------------


class NBCPT(object):
    '''
    NB Conditional Probability Table (CPT) for a child attribute.  Each child
    has only the class variable as a parent.
    '''

    def __init__(self, A_i):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT

        Params:
          - A_i: the index of the child variable

        '''
        self.cpt = np.zeros((2, 2))
        # [ [P(Ai = 0 | C = 0), P(Ai = 0 | C = 1)], [P(Ai = 1 | C = 0), P(Ai = 1 | C = 1)] ]
        self.idx = A_i

    def learn(self, A, C):
        '''
        TODO: populate any instance variables specified in __init__ we need
        to learn the parameters for this CPT

        Params:
         - A: a (n,k) numpy array where each row is a sample of assignments
         - C: a (n,) numpy array where the elements correspond to the
           class labels of the rows in A
        Return:
         - None

        '''
        # MLE solution for CPD is #(Ai = ai, C = c) / #(C = c)
        count = np.zeros((2, 2), dtype=int)
        for i in range(C.size):
            count[A[i, self.idx], C[i]] += 1

        self.cpt = (count + alpha) / (np.sum(count, axis=0) + 2 * alpha)

    def get_cond_prob(self, entry, c):
        '''
        TODO: return the conditional probability P(A_i=a_i| C=c) for the value
        a_i and class label c specified in the example entry

        Params:
         - entry: full assignment of variables
            e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
         - c: the class
        Returns:
         - p: a scalar, the conditional probability P(A_i=a_i | C=c)

        '''
        return self.cpt[entry[self.idx], c]


class NBClassifier(object):
    '''
    NB classifier class specification.
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to self._train
        Suggestions for the attributes in the classifier:
            - self.P_c: a dictionary for the probabilities for the class variable C
            - self.cpts: a list of NBCPT objects
        '''
        self.P_c = {key: None for key in {0, 1}}
        self.cpts = [NBCPT(i) for i in range(np.size(A_train, 1))]
        self._train(A_train, C_train)

    def _train(self, A_train, C_train):
        '''
        TODO: train your NB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
        Params:
          - A_train: a (n,k) numpy array where each row is a sample of assignments
          - C_train: a (n,)  numpy array where the elements correspond to
            the class labels of the rows in A
        Returns:
         - None

        '''
        for x in self.cpts:
            x.learn(A_train, C_train)
        self.P_c[1] = sum(C_train) / C_train.size
        self.P_c[0] = 1 - self.P_c[1]

    def classify(self, entry):
        '''
        TODO: return the log probabilites for class == 0 or class == 1 as a
        tuple for the given entry

        Params:
          - entry: full assignment of variables
            e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
        Returns:
         - c_pred: the predicted label, one of {0, 1}
         - logP_c_pred: the log of the conditional probability of the label |c_pred|

        '''
        num_p_1 = self.P_c[1]
        num_p_0 = self.P_c[0]
        for i in range(entry.size):
            num_p_1 *= self.cpts[i].get_cond_prob(entry, 1)
            num_p_0 *= self.cpts[i].get_cond_prob(entry, 0)

        if num_p_1 > num_p_0:
            c_pred = 1
            logP_c_pred = np.log(num_p_1 / (num_p_1 + num_p_0))
        else:
            c_pred = 0
            logP_c_pred = np.log(num_p_0 / (num_p_1 + num_p_0))

        return (c_pred, logP_c_pred)


# --------------------------------------------------------------------------
# TANB CPT and classifier
# --------------------------------------------------------------------------
class TANBCPT(object):
    '''
    TANB CPT for a child attribute.  Each child can have one other attribute
    parent (or none in the case of the root), and the class variable as a
    parent.
    '''

    def __init__(self, A_i, A_p):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the learned parameters for this CPT

        Params:
         - A_i: the index of the child variable
         - A_p: the index of its parent variable (in the Chow-Liu algorithm,
           the learned structure will have up to a single parent for each child)

        '''
        # [Ai, C, Ap]
        self.cpt = np.zeros((2, 2, 2))

        self.idx = A_i
        self.pa_idx = A_p

    def learn(self, A, C):
        '''
        TODO: populate any instance variables specified in __init__ we need to learn
        the parameters for this CPT

        Params:
         - A: a (n,k) numpy array where each row is a sample of assignments
         - C: a (n,)  numpy array where the elements correspond to the class
           labels of the rows in A
        Returns:
         - None

        '''

        if self.pa_idx != None:
            count = np.zeros((2, 2, 2), dtype=int)
            for i in range(C.size):
                count[A[i, self.idx], C[i], A[i, self.pa_idx]] += 1
        else:
            count = np.zeros((2, 2), dtype=int)
            for i in range(C.size):
                count[A[i, self.idx], C[i]] += 1

        self.cpt = (count + alpha) / (np.sum(count, axis=0) + 2 * alpha)

    def get_cond_prob(self, entry, c):
        '''
        TODO: return the conditional probability P(A_i | Pa(A_i)) for the values
        specified in the example entry and class label c
        Note: in the expression above, the class C is also a parent of A_i!

        Params;
            - entry: full assignment of variables
              e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
            - c: the class
        Returns:
         - p: a scalar, the conditional probability P(A_i | Pa(A_i))

        '''
        p = self.cpt[entry[self.idx], c, entry[self.pa_idx]
                     ] if self.pa_idx != None else self.cpt[entry[self.idx], c]

        return p


class TANBClassifier(NBClassifier):
    '''
    TANB classifier class specification
    '''

    def __init__(self, A_train, C_train):
        '''
        TODO: create any persistent instance variables you need that hold the
        state of the trained classifier and populate them with a call to self._train

        Params:
          - A_train: a (n,k) numpy array where each row is a sample of assignments
          - C_train: a (n,)  numpy array where the elements correspond to
            the class labels of the rows in A

        '''
        self.P_c = {key: None for key in {0, 1}}
        self.cpts = [None for i in range(np.size(A_train, 1))]
        # build mst, store root and edges
        mst = get_mst(A_train, C_train)
        self.root = get_tree_root(mst)
        self.cpts[self.root] = TANBCPT(self.root, None)
        for parent, child in get_tree_edges(mst, self.root):
            self.cpts[child] = TANBCPT(child, parent)

        self._train(A_train, C_train)

    def _train(self, A_train, C_train):
        '''
        TODO: train your TANB classifier with the specified data and class labels
        hint: learn the parameters for the required CPTs
        hint: you will want to look through and call functions imported from tree.py:
            - get_mst(): build the mst from input data
            - get_tree_root(): get the root of a given mst
            - get_tree_edges(): iterate over all edges in the rooted tree.
              each edge (a,b) => a -> b

        Params:
          - A_train: a (n,k) numpy array where each row is a sample of assignments
          - C_train: a (n,)  numpy array where the elements correspond to
            the class labels of the rows in A
        Returns:
         - None

        '''
        return super()._train(A_train, C_train)

    def classify(self, entry):
        '''
        TODO: return the log probabilites for class == 0 and class == 1 as a
        tuple for the given entry

        Params:
         - entry: full assignment of variables
            e.g. entry = np.array([0,1,1,...]) means variable A_0 = 0, A_1 = 1, A_2 = 1, etc.
        Returns:
         - c_pred: the predicted label in {0, 1}
         - logP_c_pred: the log conditional probability of predicting the label |c_pred|

        NOTE: this class inherits from NBClassifier, and optionally, it is possible to
        write this method in NBClassifier, such that this implementation can
        be removed.

        '''
        return super().classify(entry)


# =========================================================================


# load all data
A_base, C_base = load_vote_data()


def evaluate(classifier_cls, train_subset=False):
    '''
    =======* DO NOT MODIFY this function *=======

    evaluate the classifier specified by classifier_cls using 10-fold cross
    validation
    Params:
     - classifier_cls: either NBClassifier or TANBClassifier
     - train_subset: train the classifier on a smaller subset of the training
      data
    Returns:
     - accuracy as a proportion
     - total number of predicted samples

    '''
    global A_base, C_base

    A, C = A_base, C_base

    # score classifier on specified attributes, A, against provided labels, C
    def get_classification_results(classifier, A, C):
        results = []
        pp = []
        for entry, c in zip(A, C):
            c_pred, _ = classifier.classify(entry)
            results.append((c_pred == c))
            pp.append(_)
        return results

    # # inspect tree.py
    # mst = get_mst(A, C)
    # root = get_tree_root(mst)
    # print("root: ", root)
    # count = 0
    # for edge in get_tree_edges(mst, root):
    #     count += 1
    #     print(edge)
    # print("# of edges: ", count)

    # partition train and test set for 10 rounds
    M, N = A.shape
    tot_correct = 0
    tot_test = 0
    step = M // 10
    for holdout_round, i in enumerate(range(0, M, step)):
        A_train = np.vstack([A[0:i, :], A[i+step:, :]])
        C_train = np.hstack([C[0:i], C[i+step:]])
        A_test = A[i:i+step, :]
        C_test = C[i:i+step]
        if train_subset:
            A_train = A_train[:16, :]
            C_train = C_train[:16]

        # train the classifiers
        classifier = classifier_cls(A_train, C_train)

        train_results = get_classification_results(
            classifier, A_train, C_train)
        test_results = get_classification_results(classifier, A_test, C_test)
        tot_correct += sum(test_results)
        tot_test += len(test_results)

    return 1.*tot_correct/tot_test, tot_test


def evaluate_incomplete_entry(classifier_cls):
    '''
    TODO: Fill out the function to compute marginal probabilities.

    Params:
     - classifier_cls: either NBClassifier or TANBClassifier
     - train_subset: train the classifier on a smaller subset of the training
      data
    Returns:
     - P_c_pred: P(C = 1 | A_observed) as a scalar.
     - PA_12_eq_1: P(A_12 = 1 | A_observed) as a scalar.

    '''
    global A_base, C_base

    # train a TANB classifier on the full dataset
    classifier = classifier_cls(A_base, C_base)

    # load incomplete entry 1
    entry = load_incomplete_entry()

    c_pred, logP_c_pred = classifier.classify(entry)
    P_c_pred = np.exp(logP_c_pred)
    print('  P(C={}|A_observed) = {:2.4f}'.format(c_pred, P_c_pred))

    # TODO: write code to compute this!
    PA_12_eq_1 = None

    return P_c_pred, PA_12_eq_1


def main():
    '''
    (optional) TODO: modify or add calls to evaluate your implemented classifiers.
    '''

    # Part (a)
    print('Naive Bayes')
    accuracy, num_examples = evaluate(NBClassifier, train_subset=False)
    print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
        accuracy, num_examples))

    # Part (b)
    print('TANB Classifier')
    accuracy, num_examples = evaluate(TANBClassifier, train_subset=False)
    print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
        accuracy, num_examples))

    # # Part (c)
    # print('Naive Bayes Classifier on missing data')
    # evaluate_incomplete_entry(NBClassifier)

    # print('TANB Classifier on missing data')
    # evaluate_incomplete_entry(TANBClassifier)

    # # Part (d)
    # print('Naive Bayes')
    # accuracy, num_examples = evaluate(NBClassifier, train_subset=True)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))

    # print('TANB Classifier')
    # accuracy, num_examples = evaluate(TANBClassifier, train_subset=True)
    # print('  10-fold cross validation total test accuracy {:2.4f} on {} examples'.format(
    #     accuracy, num_examples))


if __name__ == '__main__':
    main()
