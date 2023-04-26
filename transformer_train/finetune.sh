python finetune.py \
    --lora_rank 8 \
    --num_train_epochs 5 \
    --save_steps 1000 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir ./saved_models \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 16