from .taide_dbpedia import taide_dbpedia
from .taide_piqa_en import taide_piqa_en
from .taide_piqa_tw import taide_piqa_tw
from .taide_piqa_en2tw import taide_piqa_en2tw
from .taide_piqa_tw2en import taide_piqa_tw2en
from .taide_siqa_en import taide_siqa_en
from .taide_siqa_tw import taide_siqa_tw
from .taide_ai2_arc_en import taide_ai2_arc_en
from .taide_ai2_arc_tw import taide_ai2_arc_tw
from .taide_boolq_en import taide_boolq_en
from .taide_boolq_tw import taide_boolq_tw
from .taide_openbookqa_en import taide_openbookqa_en
from .taide_openbookqa_tw import taide_openbookqa_tw
from .taide_truthful_qa_mc_en import taide_truthful_qa_mc_en
from .taide_truthful_qa_mc_tw import taide_truthful_qa_mc_tw
from .taide_winogrande_en import taide_winogrande_en
from .taide_winogrande_tw import taide_winogrande_tw
from .taide_logiqa_en import taide_logiqa_en
from .taide_logiqa_tw import taide_logiqa_tw
from .taide_cna_summary import taide_cna_summary
from .taide_hanlin_mcq import taide_hanlin_mcq
from .taide_reading_comprehension import taide_reading_comprehension
from .taide_dictionary_word2meaning import taide_dictionary_word2meaning
from .taide_dictionary_meaning2word import taide_dictionary_meaning2word
from .taide_task_classify import taide_task_classify
from .taide_topic_classify import taide_topic_classify
from .taide_answer_classify import taide_answer_classify


taskClass = {
  "taide_dbpedia": taide_dbpedia,
  "taide_piqa_en": taide_piqa_en,
  "taide_piqa_tw": taide_piqa_tw,
  "taide_piqa_en2tw": taide_piqa_en2tw,
  "taide_piqa_tw2en": taide_piqa_tw2en,
  "taide_siqa_en": taide_siqa_en,
  "taide_siqa_tw": taide_siqa_tw,
  "taide_ai2_arc_en": taide_ai2_arc_en,
  "taide_ai2_arc_tw": taide_ai2_arc_tw,
  "taide_boolq_en": taide_boolq_en,
  "taide_boolq_tw": taide_boolq_tw,
  "taide_openbookqa_en": taide_openbookqa_en,
  "taide_openbookqa_tw": taide_openbookqa_tw,
  "taide_truthful_qa_mc_en": taide_truthful_qa_mc_en,
  "taide_truthful_qa_mc_tw": taide_truthful_qa_mc_tw,
  "taide_winogrande_en": taide_winogrande_en,
  "taide_winogrande_tw": taide_winogrande_tw,
  "taide_logiqa_en": taide_logiqa_en,
  "taide_logiqa_tw": taide_logiqa_tw,
  "taide_cna_summary": taide_cna_summary,
  "taide_hanlin_mcq": taide_hanlin_mcq,
  "taide_reading_comprehension": taide_reading_comprehension,
  "taide_dictionary_word2meaning": taide_dictionary_word2meaning,
  "taide_dictionary_meaning2word": taide_dictionary_meaning2word,
  "taide_task_classify": taide_task_classify,
  "taide_topic_classify": taide_topic_classify,
  "taide_answer_classify": taide_answer_classify,
}

def construct_tasks():
  tasks = {}
  for key in taskClass.keys():
    tasks[key] = taskClass[key]
  return tasks
