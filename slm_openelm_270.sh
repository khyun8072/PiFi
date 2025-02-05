datasets=(SST2 TweetSentimentBinary)

for dataset in "${datasets[@]}"
do
    python SLM_openelm.py --dataset "$dataset" --model_name apple/OpenELM-270M --padding_side right
done