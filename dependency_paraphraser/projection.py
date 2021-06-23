import random
from functools import cmp_to_key


from dependency_paraphraser.utils import anneal, nat2ch, conll2ch


def sent2xy(sent, sentence_format='natasha'):
    pairs_x = []
    pairs_y = []
    pairs_ids = []
    tree = nat2ch(sent) if sentence_format == 'natasha' else conll2ch(sent)
    tokens = sent.tokens if sentence_format == 'natasha' else sent

    def tok2rel(tok):
        if sentence_format == 'natasha':
            return tok.rel
        return tok.deprel

    def tok2pos(tok):
        if sentence_format == 'natasha':
            return tok.pos
        return tok.upos

    def tok2features(tok, idx):
        dep = 'parent' if idx == parent_id else tok2rel(tok)
        return {
            'deprel_{}'.format(dep): 1,
            'pos_{}'.format(tok2pos(tok)): 1,
        }

    for parent_id, children_ids in tree.items():
        ids = children_ids if parent_id == -1 else [parent_id] + children_ids
        for l_id in ids:
            for r_id in ids:
                if l_id == r_id:
                    continue
                pairs_x.append({
                    '{}_{}'.format(lr, k): v
                    for lr, idx in [('r', r_id), ('l', l_id)]
                    for k, v in tok2features(tokens[idx], idx).items()
                })
                pairs_y.append(int(l_id < r_id))
                pairs_ids.append((l_id, r_id))
    return pairs_x, pairs_y, pairs_ids


def find_roots(id2children):
    """ Find the set of roots - the top nodes of all trees in the graph """
    id2parent = {}
    id2root = {}

    for parent_id, children_ids in id2children.items():
        for child_id in children_ids:
            id2parent[child_id] = parent_id

    def find_root(node_id, stack=None):
        parent_id = id2parent.get(node_id)
        if parent_id in id2root:
            return id2root[parent_id]
        if parent_id == node_id or parent_id is None:
            root_id = node_id
        elif parent_id == -1:
            root_id = parent_id
        else:
            if stack is None:
                stack = set()
            if node_id in stack:  # the "tree" is cyclic
                root_id = node_id
            else:
                stack.add(node_id)
                root_id = find_root(parent_id, stack=stack)
        id2root[node_id] = root_id
        return root_id

    for child_id in id2parent:
        find_root(child_id)

    roots = set(id2root.values())
    return roots


def project_tree_randomly(id_to_children, pair_to_proba=None, root_id=-1, temperature=1, shuffle=True):
    """
    Project a tree into a sequence, optionally with a random order of children.
    Return the list of token indices.
    """
    if root_id not in id_to_children:
        if root_id == -1:  # a sentence with empty root and no children is empty
            return []
        return [root_id]
    children_ids = [c for c in id_to_children[root_id] if c != root_id]

    ids = children_ids if root_id == -1 else [root_id] + children_ids

    def comparator(l, r):
        raw_proba = (pair_to_proba[(l, r)] + 1 - pair_to_proba[(r, l)]) / 2
        new_proba = anneal(raw_proba, t=temperature)
        return random.random() - new_proba

    if pair_to_proba:
        random.shuffle(ids)
        ids = sorted(ids, key=cmp_to_key(comparator))
    elif shuffle:
        random.shuffle(ids)
    else:
        ids = sorted(ids)

    result = []
    for tok_id in ids:
        if tok_id == root_id:
            result.append(tok_id)
        else:
            result.extend(project_tree_randomly(
                id_to_children,
                root_id=tok_id,
                pair_to_proba=pair_to_proba,
                temperature=temperature,
                shuffle=shuffle,
            ))
    return result


def make_tree_projection(sent, model, sentence_format='natasha', temperature=1):
    px, py, pids = sent2xy(sent, sentence_format=sentence_format)
    preds = model.predict_proba(px)[:, 1]
    pair2proba = {pair: proba for pair, proba in zip(pids, preds)}
    ch = nat2ch(sent) if sentence_format == 'natasha' else conll2ch(sent)
    tokens = sent.tokens if sentence_format == 'natasha' else sent
    roots = find_roots(ch)
    # filter out possible recursion
    ch = {k: [v for v in vv if v not in roots] for k, vv in ch.items()}
    positions = [
        t
        for root_id in roots
        for t in project_tree_randomly(ch, pair_to_proba=pair2proba, temperature=temperature, root_id=root_id)
    ]
    return [tokens[t] for t in positions]
