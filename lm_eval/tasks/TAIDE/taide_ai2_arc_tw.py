from lm_eval.base import MultipleChoiceTask
from datasets import load_dataset
import os
import json

class taide_ai2_arc_tw(MultipleChoiceTask):
    VERSION = 0

    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":["lm_eval/datasets/TAIDE/ai2_arc_ARC-Challenge_validation_zh-tw_gc.csv"]})
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
        ch = json.loads(doc["trans_choices"].replace("'","\""))
        choices = ch["text"]
        label = ch["label"]
        ans = doc["answerKey"]

        gold = label.index(ans)
        goal = doc["trans_question"]
        for (i,c) in enumerate(choices):
            choices[i] = f"({label[i]}) {c}"
            goal += f"\n{choices[i]}"

        out_doc = {
            "goal": goal,
            "choices": choices,
            "gold": gold,
        }
        return out_doc

    def doc_to_text(self, doc):
        return "Question: " + doc["goal"] + "\nAnswer:"
