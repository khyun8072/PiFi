TASK_LIST=(classification)
LLM=llama3.1
MODEL_LIST=(bert)
CLASSIFICATION_TASK_LIST=(sst2 imdb tweet_sentiment_binary tweet_offensive cola)

for TASK in ${TASK_LIST[@]}; do
    if [ $TASK == 'classification' ]; then
        for CLASSIFICATION_TASK in ${CLASSIFICATION_TASK_LIST[@]}; do
            for MODEL in ${MODEL_LIST[@]}; do
                    python main.py --task $TASK --job=preprocessing --task_dataset=$CLASSIFICATION_TASK --model_type=$MODEL --seed=$SEED
                    python main.py --task $TASK --job=training --task_dataset=$CLASSIFICATION_TASK --test_dataset=$CLASSIFICATION_TASK --model_type=$MODEL --method=base
                    python main.py --task $TASK --job=testing --task_dataset=$CLASSIFICATION_TASK --test_dataset=$CLASSIFICATION_TASK --model_type=$MODEL --method=base
                    python main.py --task $TASK --job=training --task_dataset=$CLASSIFICATION_TASK --test_dataset=$CLASSIFICATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
                    python main.py --task $TASK --job=testing --task_dataset=$CLASSIFICATION_TASK --test_dataset=$CLASSIFICATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
            done
        done
    fi
done
