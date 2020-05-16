import math

from collections import defaultdict


def conll2ch(sentence):
    id2children = defaultdict(list)
    for tok in sentence:
        id2children[int(tok.head) - 1].append(int(tok.id)-1)
    return id2children


def nat2ch(sentence):
    id2idx = {token.id: i for i, token in enumerate(sentence.tokens)}
    id2children = defaultdict(list)
    for tok in sentence.tokens:
        id2children[id2idx.get(tok.head_id, -1)].append(id2idx[tok.id])
    return id2children


def anneal(p, t=1):
    """ Modify a probability by applying temperature to it. """
    if t == 1:
        return p
    if t == math.inf:
        return 0.5
    if t == 0:
        return int(p > 0.5)
    return 1 / (1 + math.exp(-(math.log(p / (1-p)) / t)))
