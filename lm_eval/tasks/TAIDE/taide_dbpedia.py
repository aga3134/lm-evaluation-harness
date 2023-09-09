from lm_eval.base import Task, rf
from lm_eval import metrics
from sentence_transformers import SentenceTransformer, util
import jieba


class taide_dbpedia(Task):
    VERSION = 0
    DATASET_PATH = "TLLM/dbpedia_taiwan"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.dim = 384
        self.st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


    def has_training_docs(self):
        return True

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
        return "請簡答下列問題:\n"+doc["question"]+"\n答案:\n"

    def doc_to_target(self, doc):
        target = doc["answer"]
        return " " + target

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        #similarity
        ansEmbedding = self.st.encode(doc["answer"])
        resultEmbedding = self.st.encode(results)
        score = util.cos_sim(ansEmbedding, resultEmbedding)[0][0].item()

        #中文需要斷詞
        ref = [" ".join(jieba.cut(doc["answer"].strip()))]
        res = [" ".join(jieba.cut(results[0].strip()))]
        
        return {
            "similarity": score,
            "bleu": (ref,res),  #bleu score是針對整個corpus計算，所以留到後面的aggregation再算
        }

    def aggregation(self):
        return {
            "similarity": metrics.mean,
            "bleu": metrics.bleu,
        }

    def higher_is_better(self):
        return {
            "similarity": True,
            "bleu": True,
        }
