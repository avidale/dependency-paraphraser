import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

from dependency_paraphraser.projection import sent2xy


def label_natasha_sentences(texts):
    from dependency_paraphraser.natasha import Doc, segmenter, morph_tagger, syntax_parser
    sentences = []
    for text in texts:
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        doc.parse_syntax(syntax_parser)
        sentences.extend(doc.sents)
    return sentences


def label_udpipe_sentences(texts, model):
    sentences = []
    for text in texts:
        doc = model(text)
        sentences.extend(list(doc))
    return sentences


def load_conll_sentences(filenames):
    import pyconll
    sentences = []
    for fn in filenames:
        sentences.extend(list(
            pyconll.load_from_file(fn)
        ))
    return sentences


def train_projector(parsed_sentences, sentence_format='conll', output_file=None, cv=None):
    all_px = []
    all_py = []

    for sent in parsed_sentences:
        px, py, pids = sent2xy(sent, sentence_format=sentence_format)
        all_px.extend(px)
        all_py.extend(py)

    model = make_pipeline(
        DictVectorizer(sparse=True, dtype=int),
        PolynomialFeatures(include_bias=False, interaction_only=True),
        LogisticRegression(C=1, max_iter=1000),
    )
    if cv:
        print('Cross-validated ROC AUC scores:')
        print(cross_val_score(model, all_px, all_py, cv=cv, scoring='roc_auc'))
    model.fit(all_px, all_py)

    if output_file:
        with open('natasha_projector.pkl', 'wb') as f:
            pickle.dump(model, f)

    return model
