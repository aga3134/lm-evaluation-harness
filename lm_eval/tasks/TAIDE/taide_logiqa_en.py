from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
import json
import ast

class taide_logiqa_en(MultipleChoiceTask):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":["lm_eval/datasets/TAIDE/EleutherAI_logiqa_validation_zh-tw_gc.csv"]})
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
        goal = f"{doc['context']}\n{doc['question']}"
        options = ast.literal_eval(doc["options"])
        for (i,op) in enumerate(options):
            label = chr(ord("a")+i)
            options[i] = f"({label}) {op}"
            goal += f"\n{options[i]}"

        out_doc = {
            "goal": goal,
            "choices": options,
            "gold": ord(doc["label"])-ord("a"),
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"