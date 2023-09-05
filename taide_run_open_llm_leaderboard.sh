model="opt-1.3b_stage2--b-2048--e7_ft-e6"
limit="10"

#Read the argument values
while [[ "$#" -gt 0 ]]
  do
    case $1 in
      --model) model="$2"; shift;;
      --limit) limit="$2"; shift;;
    esac
    shift
done

modelPath="../${model}/"
outputPath="output/${model}/"

echo "model=$model"
echo "limit=$limit"
echo "modelPath=$modelPath"
echo "outputPath=$outputPath"

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks arc_challenge \
  --batch_size 2 \
  --write_out \
  --output_base_path $outputPath \
  --device cuda \
  --num_fewshot 25 \
  --limit $limit \

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks hellaswag \
  --batch_size 2 \
  --write_out \
  --output_base_path $outputPath \
  --device cuda \
  --num_fewshot 10 \
  --limit $limit \

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks hendrycksTest-* \
  --batch_size 2 \
  --write_out \
  --output_base_path $outputPath \
  --device cuda \
  --num_fewshot 5 \
  --limit $limit \

python3 main.py \
  --model hf-causal-experimental \
  --model_args pretrained=$modelPath,use_accelerate=True \
  --tasks truthfulqa_mc \
  --batch_size 2 \
  --write_out \
  --output_base_path $outputPath \
  --device cuda \
  --limit $limit \
