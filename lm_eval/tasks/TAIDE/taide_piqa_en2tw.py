from lm_eval.base import Task, rf
from lm_eval import metrics
from datasets import load_dataset
import jieba

class taide_piqa_en2tw(Task):
    VERSION = 0
    
    def download(self, data_dir=None, cache_dir=None, download_mode=None):
        self.dataset = load_dataset("csv", data_files={"validation":["lm_eval/datasets/TAIDE/piqa_validation_zh-tw_gc.csv"]})
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
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        content = f"{doc['goal']} (1) {doc['sol1']} (2) {doc['sol2']}"
        return f"英文:\n{content}\n\n中文:\n"

    def doc_to_target(self, doc):
        target = f"{doc['trans_goal']} (1) {doc['trans_sol1']} (2) {doc['trans_sol2']}"
        return target

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        target = self.doc_to_target(doc)
        target = " ".join(jieba.cut(target.strip()))
        results = " ".join(jieba.cut(results[0].strip()))
        pair = (target, results)
        #print(pair)
        return {
            "bleu": pair,
        }

    def aggregation(self):
        return {
            "bleu": metrics.bleu,
        }
    def higher_is_better(self):
        return {"bleu": True}
