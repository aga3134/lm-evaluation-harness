from lm_eval.base import Task, rf
from lm_eval import metrics
from sentence_transformers import SentenceTransformer, util
import jieba
from rouge import Rouge

class taide_cna_summary(Task):
    VERSION = 0
    DATASET_PATH = "TLLM/cna_summary"
    DATASET_NAME = None

    def __init__(self):
        super().__init__()
        self.dim = 384
        self.st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

        self.dataset = self.dataset["train"].train_test_split(test_size=0.1)
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
                self._training_docs = list(self.dataset["train"])
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            return self.dataset["test"]

    def doc_to_text(self, doc):
        return "請將下列文字做摘要:\n"+doc["input"]+"\n摘要:\n"

    def doc_to_target(self, doc):
        target = doc["output"]
        return " " + target

    def construct_requests(self, doc, ctx):
        return rf.greedy_until(ctx, {"until": ["\n"]})

    def process_results(self, doc, results):
        #similarity
        ansEmbedding = self.st.encode(doc["output"])
        resultEmbedding = self.st.encode(results)
        score = util.cos_sim(ansEmbedding, resultEmbedding)[0][0].item()

        #中文需要斷詞
        ref = " ".join(jieba.cut(doc["output"].strip()))
        res = " ".join(jieba.cut(results[0].strip()))
        
        rouge = Rouge()
        #模型回答空字串時會錯誤需擋掉
        rScore = rouge.get_scores(res, ref)[0] if res != "" else 0
        #print(rScore)

        return {
            "similarity": score,
            "rouge-1-f1": rScore["rouge-1"]["f"] if rScore !=0 else 0,
            "rouge-2-f1": rScore["rouge-2"]["f"] if rScore !=0 else 0,
            "rouge-l-f1": rScore["rouge-l"]["f"] if rScore !=0 else 0,
        }

    def aggregation(self):
        return {
            "similarity": metrics.mean,
            "rouge-1-f1": metrics.mean,
            "rouge-2-f1": metrics.mean,
            "rouge-l-f1": metrics.mean,
        }

    def higher_is_better(self):
        return {
            "similarity": True,
            "rouge-1-f1": True,
            "rouge-2-f1": True,
            "rouge-l-f1": True,
        }
