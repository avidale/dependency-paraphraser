import random


INFLECTABLES = {
    'NOUN': ['Number', 'Case'],
    'ADJ': ['Number', 'Case', 'Gender',], #  'Degree' ??
    'VERB': ['Number', 'VerbForm', 'Person', 'Tense', 'Mood', 'Gender'],
}
DEFAULT_INFLECTABLES = ['Number', 'Case', 'VerbForm', 'Person', 'Tense', 'Mood']  # 'Gender',


def morph_synonyms(token, w2v, morph_vocab=None, initial_k=30, k=10, threshold=0.0):
    if morph_vocab and hasattr(token, 'lemmatize'):
        # natasha tokens require lemmatization
        token.lemmatize(morph_vocab)
    text = token.lemma
    neighbours = []
    if text not in w2v:
        return neighbours
    pairs = w2v.most_similar(text, topn=initial_k)
    if not morph_vocab:
        return [pair for pair in pairs[:k] if pair[-1] >= threshold]
    for pair in pairs:
        if len(neighbours) >= k:
            break
        if pair[1] < threshold:
            break
        parses = morph_vocab.parse(pair[0])
        for parse in parses:
            if parse.normal == text:
                continue
            if parse.pos == token.pos:
                inflectables = INFLECTABLES.get(token.pos, DEFAULT_INFLECTABLES)
                word = parse.inflect({token.feats[k] for k in inflectables if k in token.feats})
                if not word:
                    continue
                if word.word == text:
                    break
                neighbours.append((word.word, pair[1]))
                break
    return neighbours


def replace_synonyms(tokens, w2v, morph_vocab=None, min_sim=0.6, p_rep=0.5):
    result = []
    for token in tokens:
        if random.random() > p_rep:
            result.append(token.text)
            continue
        neighbors = morph_synonyms(token, w2v, morph_vocab=morph_vocab, threshold=min_sim)

        if neighbors:
            result.append(random.choice(neighbors)[0])
        else:
            result.append(token.text)
    return result
