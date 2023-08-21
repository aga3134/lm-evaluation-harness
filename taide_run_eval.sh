python3 main.py \
  --model hf-causal \
  --model_args pretrained=../opt-1.3b_stage2--b-2048--e7_ft-e6/ \
  --tasks taide_dbpedia \
  --device cuda:0 \
  --num_fewshot 3 \
  --limit 50 \
  --description_dict_path desc.json \
  --write_out \
  --output_base_path ./output/
