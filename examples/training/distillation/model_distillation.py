import os
import csv
import gzip
import random
import logging
import argparse
from datetime import datetime
from typing import Optional

from sklearn.decomposition import PCA

import torch
from torch.utils.data import DataLoader

from sentence_transformers import models, losses, evaluation
from sentence_transformers import (
    LoggingHandler,
    SentenceTransformer,
    util,
    InputExample,
)
from sentence_transformers.datasets import ParallelSentencesDataset

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class DistillationDataset:
    def __init__(self):
        self.dataset_dict = {
            "datasets/AllNLI.tsv.gz": "https://sbert.net/datasets/AllNLI.tsv.gz",
            "datasets/wikipedia-en-sentences.txt.gz": "https://sbert.net/datasets/wikipedia-en-sentences.txt.gz",
            "datasets/stsbenchmark.tsv.gz": "https://sbert.net/datasets/stsbenchmark.tsv.gz",
        }
        self.download_dataset()

    def download_dataset(self):
        for path, url in self.dataset_dict.items():
            if not os.path.exists(path):
                util.http_get(url, path)

    def read_allnli(self):
        train_sentences_nli = set()
        dev_sentences_nli = set()
        with gzip.open("datasets/AllNLI.tsv.gz", "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == "dev":
                    dev_sentences_nli.add(row["sentence1"])
                    dev_sentences_nli.add(row["sentence2"])
                else:
                    train_sentences_nli.add(row["sentence1"])
                    train_sentences_nli.add(row["sentence2"])

        train_sentences_nli = list(train_sentences_nli)
        random.shuffle(train_sentences_nli)

        dev_sentences_nli = list(dev_sentences_nli)
        random.shuffle(dev_sentences_nli)
        dev_sentences_nli = dev_sentences_nli[0:5000]

        return train_sentences_nli, dev_sentences_nli

    def read_wikipedia(self):
        with gzip.open(
            "datasets/wikipedia-en-sentences.txt.gz", "rt", encoding="utf8"
        ) as fIn:
            wikipedia_sentences = [line.strip() for line in fIn]

        dev_sentences_wikipedia = wikipedia_sentences[0:5000]
        train_sentences_wikipedia = wikipedia_sentences[5000:]

        return train_sentences_wikipedia, dev_sentences_wikipedia

    def read_sts_benchmark(self):
        logging.info("Read STSbenchmark dev dataset")
        dev_samples = []
        with gzip.open("datasets/stsbenchmark.tsv.gz", "rt", encoding="utf8") as fIn:
            reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if row["split"] == "dev":
                    # Normalize score to range 0 ... 1
                    score = float(row["score"]) / 5.0
                    dev_samples.append(
                        InputExample(
                            texts=[row["sentence1"], row["sentence2"]], label=score
                        )
                    )

        return dev_samples


def prepare_models(
    teacher_model_name: str = "stsb-roberta-base-v2",
    use_layer_reduction: bool = True,
    student_model_name: Optional[str] = "nreimers/TinyBERT_L-4_H-312_v2",
):
    teacher_model = SentenceTransformer(teacher_model_name)

    if use_layer_reduction:
        student_model = SentenceTransformer(teacher_model_name)
        auto_model = student_model._first_module().auto_model
        layers_to_keep = [1, 4, 7, 10]

        logging.info(
            "Remove layers from student. Only keep these layers: {}".format(
                layers_to_keep
            )
        )

        new_layers = torch.nn.ModuleList(
            [
                layer_module
                for i, layer_module in enumerate(auto_model.encoder.layer)
                if i in layers_to_keep
            ]
        )
        auto_model.encoder.layer = new_layers
        auto_model.config.num_hidden_layers = len(layers_to_keep)
        return teacher_model, auto_model

    else:
        word_embedding_model = models.Transformer(student_model_name)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension()
        )
        student_model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model]
        )
        return teacher_model, student_model


def compute_pca(teacher_model, student_model):
    if (
        student_model.get_sentence_embedding_dimention()
        < teacher_model.get_sentence_embedding_dimension()
    ):
        logging.info(
            "Student model has fewer dimensions than the teacher. "
            "Compute PCA for down projection"
        )

        pca_dataset = DistillationDataset()
        train_sentences_nli, _ = pca_dataset.read_allnli()
        train_sentences_wikipedia, _ = pca_dataset.read_wikipedia()
        pca_sentences = train_sentences_nli[:20000] + train_sentences_wikipedia[:20000]

        pca_embeddings = teacher_model.encode(pca_sentences, convert_to_numpy=True)
        pca = PCA(n_components=student_model.get_sentence_embedding_dimention())
        pca.fit(pca_embeddings)

        # Add Dense layer to teacher that projects the embeddings down to the
        # student embedding size
        dense = models.Dense(
            in_features=teacher_model.get_sentence_embedding_dimension(),
            out_features=student_model.get_sentence_embedding_dimention(),
            bias=False,
            activation_function=torch.nn.Identity(),
        )
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca.components_))
        teacher_model.add_module("dense", dense)

        logging.info(
            "Teacher Performance with {} dimensions:".format(
                teacher_model.get_sentence_embedding_dimension()
            )
        )

    return teacher_model, student_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher-model",
        default="stsb-roberta-base-v2",
        help="HuggingFace id for teacher model",
    )
    parser.add_argument(
        "--student-model",
        default="nreimers/TinyBERT_L-4_H-312_v2",
        help="HuggingFace id for student model",
    )
    parser.add_argument(
        "--layer-reduction",
        default=True,
        help="Layer reduction uses the same architecture for both teacher and "
        "student but with fewer layers in the student model",
    )
    parser.add_argument(
        "--train-batch-size", default=512, help="training batch size"
    )
    parser.add_argument(
        "--inference-batch-size", default=512, help="inference batch size"
    )
    parser.add_argument(
        "--output_dir",
        default="output/model-distillation-"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        help="Output Directory",
    )
    args = parser.parse_args()

    teacher_model, student_model = prepare_models(
        teacher_model_name=args.teacher_model,
        student_model_name=args.student_model,
        use_layer_reduction=args.layer_reduction,
    )

    dataset = DistillationDataset()
    train_sentences_nli, dev_sentences_nli = dataset.read_allnli()
    train_sentences_wikipedia, dev_sentences_wikipedia = dataset.read_wikipedia()
    sts_devset = dataset.read_sts_benchmark()

    dev_evaluator_sts = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
        sts_devset, name="sts-dev"
    )

    logging.info("Teacher Performance:")
    dev_evaluator_sts(teacher_model)

    if not args.layer_reduction:
        teacher_model, student_model = compute_pca(
            teacher_model=teacher_model, student_model=student_model
        )
        dev_evaluator_sts(teacher_model)

    train_data = ParallelSentencesDataset(
        teacher_model=teacher_model,
        student_model=student_model,
        batch_size=args.inference_batch_size,
        use_embedding_cache=False,
    )
    train_data.add_dataset(
        [[sent] for sent in train_sentences_nli], max_sentence_length=256
    )
    train_data.add_dataset(
        [[sent] for sent in train_sentences_wikipedia], max_sentence_length=256
    )
    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=args.train_batch_size
    )

    train_loss = losses.MSELoss(model=student_model)

    dev_sentences = dev_sentences_nli + dev_sentences_wikipedia
    dev_evaluator_mse = evaluation.MSEEvaluator(
        dev_sentences, dev_sentences, teacher_model=teacher_model
    )

    # Train the student model to imitate the teacher
    student_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluation.SequentialEvaluator(
            [dev_evaluator_sts, dev_evaluator_mse]
        ),
        epochs=1,
        warmup_steps=1000,
        evaluation_steps=5000,
        output_path=args.output_path,
        save_best_model=True,
        optimizer_params={"lr": 1e-4, "eps": 1e-6},
        use_amp=True,
    )
