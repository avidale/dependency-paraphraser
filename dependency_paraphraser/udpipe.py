import ufal.udpipe
import pyconll

from dependency_paraphraser import synonyms, projection
from dependency_paraphraser.pretrained_projectors import natasha_projector, en_udpipe_projector


class Model:
    def __init__(self, path):
        """Load given model."""
        self.model = ufal.udpipe.Model.load(path)
        if not self.model:
            raise Exception("Cannot load UDPipe model from file '%s'" % path)

    def tokenize(self, text):
        """Tokenize the text and return list of ufal.udpipe.Sentence-s."""
        tokenizer = self.model.newTokenizer(self.model.DEFAULT)
        if not tokenizer:
            raise Exception("The model does not have a tokenizer")
        return self._read(text, tokenizer)

    def read(self, text, in_format):
        """Load text in the given format (conllu|horizontal|vertical) and return list of ufal.udpipe.Sentence-s."""
        input_format = ufal.udpipe.InputFormat.newInputFormat(in_format)
        if not input_format:
            raise Exception("Cannot create input format '%s'" % in_format)
        return self._read(text, input_format)

    def _read(self, text, input_format):
        input_format.setText(text)
        error = ufal.udpipe.ProcessingError()
        sentences = []

        sentence = ufal.udpipe.Sentence()
        while input_format.nextSentence(sentence, error):
            sentences.append(sentence)
            sentence = ufal.udpipe.Sentence()
        if error.occurred():
            raise Exception(error.message)

        return sentences

    def tag(self, sentence):
        """Tag the given ufal.udpipe.Sentence (inplace)."""
        self.model.tag(sentence, self.model.DEFAULT)

    def parse(self, sentence):
        """Parse the given ufal.udpipe.Sentence (inplace)."""
        self.model.parse(sentence, self.model.DEFAULT)

    def write(self, sentences, out_format):
        """Write given ufal.udpipe.Sentence-s in the required format (conllu|horizontal|vertical)."""

        output_format = ufal.udpipe.OutputFormat.newOutputFormat(out_format)
        output = ''
        for sentence in sentences:
            output += output_format.writeSentence(sentence)
        output += output_format.finishDocument()

        return output

    def __call__(self, text):
        sentences = self.tokenize(text)
        for s in sentences:
            self.tag(s)
            self.parse(s)
        conllu = self.write(sentences, "conllu")
        result = pyconll.load_from_string(conllu)
        return result


def paraphrase(
        text,
        udpipe_model,
        tree_temperature=0.5,
        w2v=None,
        min_sim=0.5,
        p_rep=0.5,
        projector=natasha_projector,
        morph=None,
):
    doc = udpipe_model(text)
    results = []
    for sent in doc:
        toks = projection.make_tree_projection(
            sent, model=projector, temperature=tree_temperature, sentence_format='conll',
        )
        if w2v:
            words = synonyms.replace_synonyms(
                toks, w2v=w2v, morph_vocab=morph, min_sim=min_sim, p_rep=p_rep,
            )
        else:
            words = [token.form for token in toks]
        results.append(' '.join(words))
    return ' '.join(results)
