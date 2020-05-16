from dependency_paraphraser.pretrained_projectors import natasha_projector


from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    Doc,
)

from dependency_paraphraser import projection, synonyms


segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)

gensim_emb = None


def use_news_embeddings():
    """ Convert navec embeddings to gensim format to use for synonym replacement """
    global gensim_emb
    gensim_emb = emb.as_gensim


def paraphrase(text, tree_temperature=0.5, w2v=None, min_sim=0.5, p_rep=0.5, projector=natasha_projector):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    if w2v is None:
        w2v = gensim_emb

    results = []

    for sent in doc.sents:
        toks = projection.make_tree_projection(
            sent, model=projector, temperature=tree_temperature,
        )
        if w2v:
            words = synonyms.replace_synonyms(
                toks, w2v=w2v, morph_vocab=morph_vocab, min_sim=min_sim, p_rep=p_rep,
            )
        else:
            words = [token.text for token in toks]
        results.append(' '.join(words))
    return ' '.join(results)
