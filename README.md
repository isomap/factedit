# Fact-based Text Editing
[![Conference](https://img.shields.io/badge/acl-2020-red)](https://www.aclweb.org/anthology/2020.acl-main.17/)
[![arXiv](https://img.shields.io/badge/arxiv-2007.00916-success)](https://arxiv.org/abs/2007.00916/)
[![Slide](https://img.shields.io/badge/slide-pdf-informational)](https://isomap.github.io/slides/iso2020acl.pdf)

<p align="center"><img width="60%" src="img/task.png"/></p>


Code and Datasets for [Fact-based Text Editing (Iso et al; ACL 2020)](https://www.aclweb.org/anthology/2020.acl-main.17/).

## Dataset
<p align="center"><img width="60%" src="img/insertion.png"/><img width="60%" src="img/deletion.png"/></p>

Datasets are created from publicly availlable table-to-text datasets.
The dataset created from ["webnlg"](https://github.com/ThiagoCF05/webnlg/) referred to as "webedit", and the dataset created from ["rotowire(-modified)"](https://github.com/aistairc/rotowire-modified) referred to as the "rotoedit" data.


To extract the data, run `tar -jxvf webedit.tar.bz2` to form a webedit/ directory (and similarly for rotoedit.tar.bz2).

## Model overview
<p align="center"><img width="60%" src="img/model.png"/></p>

The model, which we call **FactEditor**, consists of three components, a buffer for storing the draft text and its representations, a stream for storing the revised text and its representations, and a triples for storing the triples and their representations.

FactEditor scans the text in the buffer, copies the parts of text from the buffer into the stream if they are described in the triples in the memory, deletes the parts of the text if they are not mentioned in the triples, and inserts new parts of next into the stream which is only presented in the triples.


## Usage

### Dependencies
- The code was written for Python 3.X and requires [AllenNLP](https://allennlp.org/).
- Dependencies can be installed using `requirements.txt`.

### Training
Set your config file path and serialization dirctory as environment variables: 
```
export CONFIG=<path to the config file>
export SERIALIZATION_DIR=<path to the serialization_dir>
```

Then you can train FactEditor:
```python
allennlp train $CONFIG \
            -s $SERIALIZATION_DIR \
            --include-package editor
```

For example, the following is the sample script for training the model with WebEdit dataset:
```python
allennlp train config/webedit.jsonnet \
            -s models/webedit \
            --include-package editor 
```

### Decoding
Set the dataset you want to decode and the model checkpoint you want to use as environment variables:
```
export INPUT_FILE=<path to the dev/test file>
export ARCHIVE_FILE=<path to the model archive file>
```

Then you can decode with FactEditor:
```python
python predict.py $INPUT_FILE \
                  $ARCHIVE_FILE \
                  --cuda_device -1
```
To run on a GPU, run with `--cuda_device 0` (or any other CUDA devices).

To run the model with a pretrained checkpoint the development set of WebEdit data:
```python
python predict.py ./data/webedit/dev.jsonl \
                  ./models/webedit.tar.gz \
                  --cuda_device -1
```

## References
```tex
@InProceedings{iso2020fact,
    author = {Iso, Hayate and
              Qiao, Chao and
              Li, Hang},
    title = {Fact-based Text Editing},
    booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
    pages={171--182},
    year = {2020}
  }
```
