import os
from glob import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,6'

import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import argparse

os.environ['WANDB_DISABLED'] = 'true'
os.environ['COMET_AUTO_LOG_DISABLE'] = 'true'
transformers.logging.set_verbosity_info()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
        
    train_filenames_mask = 'data/train_shuffle_descriptions_mlm_v2.txt'
    val_filenames_mask = 'data/val_shuffle_descriptions_mlm_v2.txt'
    pretrained_path = 'xlm-roberta-large'
    check_path = 'outputs/checkpoint-19500'
    max_length = 256
    batch_size = 8
    gradient_accumulation_steps = 1
    log_dir = 'outputs/'

    os.makedirs(log_dir, exist_ok=True)
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    if local_rank in {0, -1}:
        tokenizer.save_pretrained(os.path.join(log_dir, 'model', 'tokenizer'))

    model = AutoModelForMaskedLM.from_pretrained(check_path)
    model.tie_weights()
    model = model.to(device)
    # model = torch.compile(model)

    datasets = load_dataset(
        'text',
        data_files={
            'train': [train_filenames_mask],
            'val': [val_filenames_mask]
        }
    )

    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]


    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
        return tokenizer(
            examples["text"],
            # padding='max_length',
            padding=False,
            truncation=True,
            max_length=max_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )


    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count(),
        remove_columns=[text_column_name],
        load_from_cache_file=True
    )

    data_collator = DataCollatorForWholeWordMask(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )


    training_args = TrainingArguments(
        output_dir=log_dir,
        overwrite_output_dir=True,
        report_to='tensorboard',
        num_train_epochs=40,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=2e-5 / 32 * batch_size * gradient_accumulation_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=10,
        local_rank=local_rank,
        fp16=False,
        ddp_find_unused_parameters=False,
        dataloader_pin_memory=True,
        prediction_loss_only=True,
        evaluation_strategy=transformers.trainer_utils.IntervalStrategy.EPOCH,
        save_strategy=transformers.trainer_utils.IntervalStrategy.EPOCH,
        load_best_model_at_end=True,
        save_total_limit=3,
        dataloader_num_workers=os.cpu_count() // 2,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val']
    )
    trainer.train()
    trainer.save_model(os.path.join(log_dir, 'model'))
