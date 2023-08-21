from .taide_dbpedia import taide_dbpedia

taskClass = {
  "dbpedia": taide_dbpedia,
}

def construct_tasks():
  tasks = {}
  for key in taskClass.keys():
    tasks["taide_"+key] = taskClass[key]
  return tasks
