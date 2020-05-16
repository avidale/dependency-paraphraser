# dependency-paraphraser
A sentence paraphraser based on dependency parsing 
and word embedding similarity.

How the paraphraser works:
1. Create a random projection of the dependency tree 
2. Replace several words with similar ones

The basic usage (for Russian language) is based on Natasha library:

```
pip install dependency-paraphraser natasha
```

```python
import dependency_paraphraser.natasha
import random
random.seed(42)
text = 'каждый охотник желает знать где сидит фазан'
for i in range(3):
    print(dependency_paraphraser.natasha.paraphrase(text, tree_temperature=2))
# желает знать сидит фазан где каждый охотник
# каждый охотник желает знать где фазан сидит
# знать где фазан сидит каждый охотник желает
```

You can provide your own w2v model to replace words with similar ones:
```python
import compress_fasttext
small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
    'https://github.com/avidale/compress-fasttext/releases/download/v0.0.1/ft_freqprune_100K_20K_pq_100.bin'
)
random.seed(42)
for i in range(3):
    print(dependency_paraphraser.natasha.paraphrase(text, w2v=small_model, p_rep=0.8, min_sim=0.55))
# стремится каждый охотник знать рябчик где усаживается
# каждый охотник хочет узнать фазан где просиживает
# каждый охотник хочет узнать фазан где восседает
```

Alternatively, you can expand and use the w2v model from Natasha (aka `navec`):
```python
navec_model = dependency_paraphraser.natasha.emb.as_gensim
random.seed(42)
for i in range(3):
    print(dependency_paraphraser.natasha.paraphrase(text, w2v=navec_model, p_rep=0.5, min_sim=0.55))
# желает каждый охотник помнить фазан где лежит
# каждый охотник желает знать фазан где сидит
# каждый охотник оставляет понять где фазан лежит
```

For other languages, one way to use this paraphraser is with the UDPipe library

```
pip install dependency-paraphraser ufal.udpipe pyconll
```

```python
import dependency_paraphraser.udpipe
path = 'english-ewt-ud-2.5-191206.udpipe'
pipe = dependency_paraphraser.udpipe.Model(path)
projector = dependency_paraphraser.udpipe.en_udpipe_projector

text = 'in April 2012 they released the videoclip for a new single entitled Giorgio Mastrota'
for i in range(3):
    print(dependency_paraphraser.udpipe.paraphrase(text, pipe, projector=projector, tree_temperature=1))
# they released the videoclip in April 2012 for a new entitled Mastrota single Giorgio
# they released in April 2012 the videoclip for a entitled single new Giorgio Mastrota
# they released the videoclip in April 2012 for a new single Giorgio Mastrota entitled
```

Projectors (models for projecting dependency trees into a flat sentence)
can be trained for any language, if you have a corpus of unlabeled sentences
and a syntax parser to label them:

```python
import dependency_paraphraser.udpipe
import dependency_paraphraser.train_projector
parser = dependency_paraphraser.udpipe.Model(path_to_your_model)

sents = dependency_paraphraser.train_projector.label_udpipe_sentences(
    texts=your_corpus,
    model=parser,
)
projector = dependency_paraphraser.train_projector.train_projector(sents)
```
