# Instructions to quickly generate data

This is a modified version of the original Pareto Probing repository.

1. Create a conda environment with ```conda env create -f environment.yml```
2. Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```
3.  Install the newest version of fastText:
```bash
$ pip install git+https://github.com/facebookresearch/fastText
```
4. Download the UD treebanks
```bash
$ make get_ud
```
5. Preprocess the treebanks. Need to run this for each language you want (see options in `src/util/ud\_list.py`),
```bash
$ make process LANGUAGE=<language> REPRESENTATION=ud
```
6. Get the embeddings for each sentence for whatever representation you want (e.g., `bert`). Need to run this for each language you want (see options in `src/util/ud\_list.py`),
```bash
$ make process LANGUAGE=<language> REPRESENTATION=<representation>
```
7. Run the following command for the desired language/representation pairs. This will generate files that can be directly plugged into the bayesian probing codebase.
```bash
python -u src/h02_learn/convert_data.py --language <language> --representation <representation>
```


# pareto-probing

This repository contains code accompanying the paper: "Pareto Probing: Trading Off Accuracy and Complexity" (Pimentel et al., EMNLP 2020). It is a study of probing in the context of a Pareto trade-off.

## Install Dependencies

Create a conda environment with
```bash
$ conda env create -f environment.yml
```
Then activate the environment and install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).
```bash
$ conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
$ # conda install pytorch torchvision cpuonly -c pytorch
$ pip install transformers
```
Install the newest version of fastText:
```bash
$ pip install git+https://github.com/facebookresearch/fastText
```

## Download and parse universal dependencies (UD) data

You can easily download UD data with the following command
```bash
$ make get_ud
```

You can then get the embeddings for it with command
```bash
$ make process LANGUAGE=<language> REPRESENTATION=<representation>
```

This repository has the option of using representations: onehot; random; bert; albert; and roberta.
As languages, you should be able to experiment on: 'en' (english); 'cs' (czech); 'eu' (basque); 'fi' (finnish); 'tr' (turkish); 'ar' (arabic); 'ja' (japanese); 'ta' (tamil); 'ko' (korean); 'mr' (marathi); 'ur' (urdu); 'te' (telugu); 'id' (indonesian).
If you wanna experiment on other languages, add the appropriate language code to `src/util/constants.py` and the ud path to `src/util/ud_list.py`.


## Train your models

You can train your models using random search with the command
```bash
$ make train LANGUAGE=<language> REPRESENTATION=<representation> TASK=<task> MODEL=<model>
```
There are three tasks available in this repository: pos_tag; dep_label; and parse.
We also have three models available: 'mlp'; 'linear'; and 'max-rank'.


## Extra Information

#### Citation

If this code or the paper were usefull to you, consider citing it:


```bash
@inproceedings{pimentel-etal-2020-pareto,
    title = "Pareto Probing: {T}rading Off Accuracy and Complexity",
    author = "Pimentel, Tiago and
    Saphra, Naomi and
    Williams, Adina and
    Cotterell, Ryan",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year = "2020",
    publisher = "Association for Computational Linguistics",
}
```


#### Contact

To ask questions or report problems, please open an [issue](https://github.com/rycolab/pareto-probing/issues).
