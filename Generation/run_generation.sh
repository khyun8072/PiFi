clear

TASK_LIST=(summarization )

MODEL_LIST=(t5)
LLM=llama3.1



SUMMARIZATION_TASK=cnn_dailymail
TRANSLATION_TASK=multi30k

for TASK in ${TASK_LIST[@]}; do
    if [ $TASK == 'summarization' ]; then
        for MODEL in ${MODEL_LIST[@]}; do
            python main.py --task $TASK --job=preprocessing --task_dataset=$SUMMARIZATION_TASK --model_type=$MODEL
            python main.py --task $TASK --job=training --task_dataset=$SUMMARIZATION_TASK --model_type=$MODEL --method=base
            python main.py --task $TASK --job=testing --task_dataset=$SUMMARIZATION_TASK --model_type=$MODEL --method=base
            python main.py --task $TASK --job=training --task_dataset=$SUMMARIZATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
            python main.py --task $TASK --job=testing --task_dataset=$SUMMARIZATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
        done
    elif [ $TASK == 'translation' ]; then
        for MODEL in ${MODEL_LIST[@]}; do
            python main.py --task $TASK --job=preprocessing --task_dataset=$TRANSLATION_TASK --model_type=$MODEL
            python main.py --task $TASK --job=training --task_dataset=$TRANSLATION_TASK --model_type=$MODEL --method=base
            python main.py --task $TASK --job=testing --task_dataset=$TRANSLATION_TASK --model_type=$MODEL --method=base
            python main.py --task $TASK --job=training --task_dataset=$TRANSLATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
            python main.py --task $TASK --job=testing --task_dataset=$TRANSLATION_TASK --model_type=$MODEL --method=base_llm --llm=$LLM
        done
    fi
done

