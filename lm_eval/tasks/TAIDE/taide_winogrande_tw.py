from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
import json

class taide_winogrande_tw(MultipleChoiceTask):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":["lm_eval/datasets/TAIDE/winogrande_winogrande_debiased_validation_zh-tw_gc.csv"]})
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
        goal = f"哪個句子比較合理?\n"
        for i in range(2):
            index = str(i+1)
            doc["trans_sentence"+index] = f"({index}) "+doc["trans_sentence"+index]
            goal += doc["trans_sentence"+index]+"\n"

        out_doc = {
            "goal": goal,
            "choices": [doc["trans_sentence"+str(i+1)] for i in range(2)],
            "gold": int(doc["answer"])-1,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "問題: " + doc["goal"] + "\n答案:"
