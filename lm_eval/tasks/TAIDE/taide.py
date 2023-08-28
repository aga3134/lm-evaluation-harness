from .taide_dbpedia import taide_dbpedia
from .taide_piqa_en import taide_piqa_en
from .taide_piqa_tw import taide_piqa_tw
from .taide_piqa_en2tw import taide_piqa_en2tw
from .taide_piqa_tw2en import taide_piqa_tw2en
from .taide_siqa_en import taide_siqa_en
from .taide_siqa_tw import taide_siqa_tw
from .taide_ai2_arc_en import taide_ai2_arc_en
from .taide_ai2_arc_tw import taide_ai2_arc_tw

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
}

def construct_tasks():
  tasks = {}
  for key in taskClass.keys():
    tasks[key] = taskClass[key]
  return tasks
