#避免bash把taide_*展開成taide_開頭的檔案(noglob)
set -f

task="taide_*"
model="opt-1.3b_stage2--b-2048--e7_ft-e6"
fewshot=3
limit="10"

#Read the argument values
while [[ "$#" -gt 0 ]]
  do
    case $1 in
      --task) task="$2"; shift;;
      --model) model="$2"; shift;;
      --fewshot) fewshot="$2"; shift;;
      --limit) limit="$2"; shift;;
    esac
    shift
done

modelPath="../${model}/"
outputPath="output/${model}/"

echo "task=$task"
echo "model=$model"
echo "fewshot=$fewshot"
echo "limit=$limit"
echo "modelPath=$modelPath"
echo "outputPath=$outputPath"

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks $task \
  --device cuda:0 \
  --num_fewshot $fewshot \
  --description_dict_path desc.json \
  --write_out \
  --output_base_path $outputPath \
  --limit $limit
