TASK_LIST=(question_answering)
MODEL_LIST=(bert roberta electra deberta debertav3)
LLM=llama3.1

QUESTION_ANSWERING_TASK_LIST=(squad)

# 실험 1-a
for TASK in ${TASK_LIST[@]}; do
    if [ $TASK == 'question_answering' ]; then
        for QUESTION_ANSWERING_TASK in ${QUESTION_ANSWERING_TASK_LIST[@]}; do
            for MODEL in ${MODEL_LIST[@]}; do
                python main.py --task $TASK --job=preprocessing --task_dataset=$QUESTION_ANSWERING_TASK --model_type=$MODEL
                python main.py --task $TASK --job=training --task_dataset=$QUESTION_ANSWERING_TASK --model_type=$MODEL --method=base
                python main.py --task $TASK --job=testing --task_dataset=$QUESTION_ANSWERING_TASK --test_dataset=$QUESTION_ANSWERING_TASK --model_type=$MODEL --method=base
                python main.py --task $TASK --job=training --task_dataset=$QUESTION_ANSWERING_TASK --test_dataset=$QUESTION_ANSWERING_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
                python main.py --task $TASK --job=testing --task_dataset=$QUESTION_ANSWERING_TASK --test_dataset=$QUESTION_ANSWERING_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
            done
        done
    fi
done