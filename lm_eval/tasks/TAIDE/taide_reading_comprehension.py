from lm_eval.base import Task, rf
from lm_eval import metrics
from sentence_transformers import SentenceTransformer, util
import jieba


class taide_reading_comprehension(Task):
    VERSION = 0
    DATASET_PATH = "TLLM/reading-comprehension"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.dim = 384
        self.st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')


    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

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
        return "請依下列文章回答問題:\n"+doc["prompt"]+"\n回答:"

    def doc_to_target(self, doc):
        target = doc["response"]
        return " " + target

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        #similarity
        ansEmbedding = self.st.encode(doc["response"])
        resultEmbedding = self.st.encode(results)
        score = util.cos_sim(ansEmbedding, resultEmbedding)[0][0].item()

        #中文需要斷詞
        ref = [" ".join(jieba.cut(doc["response"].strip()))]
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
