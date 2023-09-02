#避免bash把taide_*展開成taide_開頭的檔案(noglob)
set -f

task="taide_*"
if [ -n "$1" ]
  then
    task=$1
fi

model="opt-1.3b_stage2--b-2048--e7_ft-e6/"
modelPath="../${model}"
outputPath="output/${model}/"
limit="10"

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks $task \
  --device cuda:0 \
  --num_fewshot 3 \
  --description_dict_path desc.json \
  --write_out \
  --output_base_path $outputPath \
  --limit $limit
