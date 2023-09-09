from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
import json
import ast

class taide_hanlin_mcq(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "TLLM/hanlin_mcq"
    DATASET_NAME = None

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        super().download(data_dir, cache_dir, download_mode)

        self.dataset = self.dataset["train"].train_test_split(test_size=0.5)
        print(self.dataset)

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(
                    map(self._process_doc, self.dataset["train"])
                )
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        goal = doc["題目"]

        choices = []
        choices.append("\n(1) "+doc["選項一"])
        goal += choices[0]
        choices.append("\n(2) "+doc["選項二"])
        goal += choices[1]
        choices.append("\n(3) "+doc["選項三"])
        goal += choices[2]
        choices.append("\n(4) "+doc["選項四"])
        goal += choices[3]

        out_doc = {
            "goal": goal,
            "choices": choices,
            "gold": doc["正確答案"]-1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "問題: " + doc["goal"] + "\n答案:"
