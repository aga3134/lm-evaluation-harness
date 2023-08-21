python3 -m scripts.write_out \
  --output_base_path ./output \
  --tasks taide_dbpedia \
  --sets val \
  --num_fewshot 3 \
  --num_examples 5 \
  --description_dict_path desc.json
