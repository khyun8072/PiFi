datasets=(SST2 TweetSentimentBinary)


for dataset in "${datasets[@]}"
do
    python SLM_smollm.py --dataset "$dataset" --model_name HuggingFaceTB/SmolLM2-360M --padding_side right
done