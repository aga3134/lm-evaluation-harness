task="taide_ai2_arc_tw"
if [ -n "$1" ]
  then
    task=$1
fi

python3 -m scripts.write_out \
  --output_base_path ./output \
  --tasks $task \
  --sets val \
  --num_fewshot 3 \
  --num_examples 5 \
  --description_dict_path desc.json
