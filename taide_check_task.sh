task="taide_piqa_tw"
fewshot=3
output="./output"
desc="desc.json"
sets="val"

#Read the argument values
while [[ "$#" -gt 0 ]]
  do
    case $1 in
      --task) task="$2"; shift;;
      --fewshot) fewshot="$2"; shift;;
      --output) output="$2"; shift;;
      --desc) desc="$2"; shift;;
      --sets) sets="$2"; shift;;
    esac
    shift
done

echo "task=$task"
echo "fewshot=$fewshot"
echo "output=$output"
echo "desc=$desc"
echo "sets=$sets"

python3 -m scripts.write_out \
  --output_base_path $output \
  --tasks $task \
  --sets $sets \
  --num_fewshot $fewshot \
  --num_examples 5 \
  --description_dict_path $desc
