from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
import json
import ast

class taide_topic_classify(MultipleChoiceTask):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":["lm_eval/datasets/TAIDE/instruction_topic_classify.csv"]})
        print(self.dataset)

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

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
        choices = ast.literal_eval(doc["options"])
        gold = doc["label"]

        goal = "請選擇下列任務的主題:\n"+doc["prompt"]
        for (i,c) in enumerate(choices):
            choices[i] = f"({i+1}) {c}"
            goal += f"\n{choices[i]}"

        out_doc = {
            "goal": goal,
            "choices": choices,
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "問題: " + doc["goal"] + "\n回答:"