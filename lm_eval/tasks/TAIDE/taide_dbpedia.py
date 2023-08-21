from lm_eval.base import Task, rf
from lm_eval import metrics
from sentence_transformers import SentenceTransformer, util

#dim = 384
#st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

class taide_dbpedia(Task):
    VERSION = 0
    DATASET_PATH = "TLLM/dbpedia_taiwan"
    DATASET_NAME = None

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
        return "Q: "+doc["question"]

    def doc_to_target(self, doc):
        target = doc["answer"]
        return " A: " + target

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        print(results)
        ansEmbedding = st.encode(doc["answer"])
        resultEmbedding = st.encode(results)
        score = util.cos_sim(ansEmbedding, resultEmbedding)[0][0].item()
        
        return {"similarity": score}

    def aggregation(self):
        return {"similarity": metrics.mean}

    def higher_is_better(self):
        return {"similarity": True}
