from typing import Dict, List

import nltk
from allennlp.data import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.predictors.seq2seq import Seq2SeqPredictor
from allennlp.models import Model

from sari_hook import get_sari_score


def batch(data, batch_size, ignore_smaller=False):
    mini_batch = []
    for ex in data:
        mini_batch.append(ex)
        if len(mini_batch) == batch_size:
            yield mini_batch
            mini_batch = []
    if not ignore_smaller and mini_batch:
        yield mini_batch


def get_prediction(model: Model,
                   reader: DatasetReader,
                   data_path: str,
                   batch_size: int = 1024):
    model.eval()
    data = reader.read(data_path)
    predictor = Seq2SeqPredictor(model, reader)

    for ins in batch(data, batch_size):
        yield from predictor.predict_batch_instance(ins)


class Evaluator:
    def __init__(self):
        self.ref, self.hyp, self.src = [], [], []

    def __call__(self,
                 output_dict: Dict) -> None:
        self.update(output_dict)

    @staticmethod
    def get_list(xs: List[Token]):
        return list(map(str, xs))

    def update(self,
               output_dict: Dict) -> None:
        h = self.get_list(output_dict["predicted_tokens"][0])
        d = self.get_list(output_dict["metadata"]["draft"])
        r = self.get_list(output_dict["metadata"]["revised"])

        self.ref.append(r)
        self.hyp.append(h)
        self.src.append(d)

    def get_metrics(self,
                    reset: bool = True):
        sari = [get_sari_score(d, h, [r], beta_for_deletion=1.) for d, h, r in zip(self.src, self.hyp, self.ref)]
        em = [h == r for h, r in zip(self.hyp, self.ref)]

        scores = {
            "BLEU": nltk.bleu_score.corpus_bleu([[r] for r in self.ref], self.hyp),
            "EM": sum(em) / len(em)
        }
        scores.update({k: sum(v) / len(v) for k, v in zip(("SARI", "KEEP", "ADD", "DEL"), zip(*sari))})

        if reset:
            self.ref, self.hyp, self.src = [], [], []
        return scores
