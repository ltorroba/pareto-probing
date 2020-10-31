import numpy as np
import torch
from itertools import groupby

from .pos_tag import PosTagDataset


class DepLabelDatasetProbekit(PosTagDataset):
    name = 'dep_label'

    def load_index(self, x_raw, words=None):
        raise NotImplementedError("This function should not have to be called")

        if words is None:
            words = []

        new_words = sorted(list(set(np.unique(x_raw)) - set(words)))
        if new_words:
            words = np.concatenate([words, new_words])

        words_dict = {word: i for i, word in enumerate(words)}
        x = np.array([[words_dict[token] for token in tokens] for tokens in x_raw])

        self.x = torch.from_numpy(x)
        self.words = words

        self.n_words = len(words)

    def load_data(self, iterator):
        x_raw, y_raw, probekit_words = [], [], []
        for sentence_ud, sentence_tokens in iterator():
            for i, token in enumerate(sentence_ud):
                head = token['head']
                rel = token['rel']

                if rel in {"_", "root"}:
                    continue

                x_raw_tail = sentence_tokens[i]
                x_raw_head = sentence_tokens[head - 1]

                x_raw += [[x_raw_tail, x_raw_head]]
                y_raw += [rel]
                probekit_words += [token["word"]]

        x_raw = np.array(x_raw)
        y_raw = np.array(y_raw)

        # This joins each of the 2 vectors (e.g., 768-d) into a single one (e.g., 1536)
        if len(x_raw.shape) == 3:
            x_raw = x_raw.reshape(x_raw.shape[0], -1)  # pylint: disable=E1136  # pylint/issues/3139

        self._probekit_words = probekit_words

        return x_raw, y_raw

    def _process(self, classes):
        x_raw, y_raw = self.load_data(self.iterate_embeddings)

        data = []
        for embed, dep, word in zip(x_raw, y_raw, self._probekit_words):
            data += [{
                "word": word,
                "embedding": embed,
                "attributes": {
                    "dep": dep
                }
            }]


        self._probekit_data = data

        self.load_embeddings(x_raw)
        self.load_classes(y_raw, classes=classes)
