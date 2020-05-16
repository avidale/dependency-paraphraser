import os
import pickle
import sklearn  # noqa

models_path = os.path.join(os.path.dirname(__file__), 'models')

with open(os.path.join(models_path, 'natasha_projector.pkl'), 'rb') as f:
    natasha_projector = pickle.load(f)

with open(os.path.join(models_path, 'en_udpipe_projector.pkl'), 'rb') as f:
    en_udpipe_projector = pickle.load(f)
