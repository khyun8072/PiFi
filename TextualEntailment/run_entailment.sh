TASK_LIST=(entailment)
MODEL_LIST=(bert)
LLM=llama3.1
ENTAILMENT_TASK_LIST=(mnli snli)

# 실험 1-a
for TASK in ${TASK_LIST[@]}; do
    if [ $TASK == 'entailment' ]; then
        for ENTAILMENT_TASK in ${ENTAILMENT_TASK_LIST[@]}; do
            for MODEL in ${MODEL_LIST[@]}; do
                python main.py --task $TASK --job=preprocessing --task_dataset=$ENTAILMENT_TASK --model_type=$MODEL
                python main.py --task $TASK --job=training --task_dataset=$ENTAILMENT_TASK --test_dataset=$ENTAILMENT_TASK --model_type=$MODEL --method=base
                python main.py --task $TASK --job=testing --task_dataset=$ENTAILMENT_TASK --test_dataset=$ENTAILMENT_TASK --model_type=$MODEL --method=base
                python main.py --task $TASK --job=training --task_dataset=$ENTAILMENT_TASK --test_dataset=$ENTAILMENT_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
                python main.py --task $TASK --job=testing --task_dataset=$ENTAILMENT_TASK --test_dataset=$ENTAILMENT_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
            done
        done
    fi
done