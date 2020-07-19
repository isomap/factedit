import sys
import click
import json
from tqdm import tqdm

from allennlp.predictors import Predictor
from allennlp.data import DatasetReader
from allennlp.models.archival import load_archive

from editor import EditReader, Editor
from util import batch, Evaluator


@click.command()
@click.argument("input_file")
@click.argument("archive_file")
@click.option("--batch_size", type=click.INT, default=32)
@click.option("--cuda_device", type=click.INT, default=-1)
def main(input_file, archive_file, batch_size, cuda_device):
    model, config = load_archive(archive_file=archive_file, cuda_device=cuda_device)
    model.eval()

    dataset_reader = DatasetReader.from_params(config["dataset_reader"])
    dataset = dataset_reader.read(input_file)
    predictor = Predictor(model, dataset_reader)
    evaluator = Evaluator()

    with tqdm(desc="Decoding...") as p:
        for ins in batch(dataset, batch_size):
            for result in predictor.predict_batch_instance(ins):
                print(json.dumps(result))
                evaluator(result)
                p.update()
    print(evaluator.get_metrics(reset=True), file=sys.stderr)


if __name__ == '__main__':
    main()
