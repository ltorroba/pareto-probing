import sys
import os; sys.path.insert(1, os.path.join(sys.path[0], '..'))  # noqa
from pathlib import Path
import pickle
from h02_learn.dataset.dep_label_probekit import DepLabelDatasetProbekit
from argparse import ArgumentParser


parser = ArgumentParser(description="Converts the Pareto Probing data to the format used in Probekit.")
parser.add_argument("--language", type=str, help="The language to consider, e.g., 'english'.")
parser.add_argument("--representation", choices=["bert", "albert", "roberta", "fast"],
                    help="The representation to used to embed tokens.")
args = parser.parse_args()


data_path = "./data/processed"
output_path = "./data/probekit"
task = "dep_label"
language = args.language
representation = args.representation

if representation in ["bert", "albert", "roberta"]:
    embedding_size = 768
elif representation == "fast":
    embedding_size = 300

if task == "dep_label":
    embedding_size = embedding_size * 2


dataset_cls = DepLabelDatasetProbekit
train = dataset_cls(data_path, language, representation, embedding_size, "train")
classes, words = train.classes, train.words

dev = dataset_cls(data_path, language, representation, embedding_size, "dev", classes=classes, words=words)
test = dataset_cls(data_path, language, representation, embedding_size, "test", classes=classes, words=words)


Path(output_path).mkdir(parents=True, exist_ok=True)
with open(Path(output_path) / f"{task}-{language}-{representation}-train.pkl", "wb") as h:
    pickle.dump(train._probekit_data, h)

with open(Path(output_path) / f"{task}-{language}-{representation}-valid.pkl", "wb") as h:
    pickle.dump(dev._probekit_data, h)

with open(Path(output_path) / f"{task}-{language}-{representation}-test.pkl", "wb") as h:
    pickle.dump(test._probekit_data, h)
