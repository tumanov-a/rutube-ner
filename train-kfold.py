from process_data import preprocess_data
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from transformers import AutoTokenizer 
import pandas as pd
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import numpy as np
import logging
from transformers.trainer import logger as noisy_logger
from sklearn.model_selection import KFold
noisy_logger.setLevel(logging.WARNING)
import torch
import gc
    

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels, zero_division=0)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        label_ids = [label_list.index(idx) if isinstance(idx, str) else idx for idx in label_ids]

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

if __name__ == '__main__':
    all_trainers = []
    batch_size = 8
    folds = 3
    model_checkpoint = "xlm-roberta-large"  # "cointegrated/rubert-tiny"
    checkpoint_path = 'pretrain_models/checkpoint-19500'
    
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    metric = load_metric("seqeval")
    ner_data = preprocess_data("data/ner_data_train.csv")
    
    ner_val_train, ner_test = train_test_split(ner_data, test_size=0.1, random_state=1)
    kf = KFold(n_splits=folds, random_state=1, shuffle=True)
    ner_val_train = np.array(ner_val_train)
    
    for fold, (train_index, val_index) in enumerate(kf.split(ner_val_train)):
        ner_train = ner_val_train[train_index].tolist()
        ner_val = ner_val_train[val_index].tolist()
        
        label_list = sorted({label for item in ner_train for label in item['tags']})

        ner_data = DatasetDict({
            'train': Dataset.from_pandas(pd.DataFrame(ner_train)),
            'test': Dataset.from_pandas(pd.DataFrame(ner_test)),
            'val': Dataset.from_pandas(pd.DataFrame(ner_val))
        })

        tokenized_datasets = ner_data.map(tokenize_and_align_labels, batched=True)

        model = AutoModelForTokenClassification.from_pretrained(checkpoint_path, num_labels=len(label_list))
        model.config.id2label = dict(enumerate(label_list))
        model.config.label2id = {v: k for k, v in model.config.id2label.items()}

        data_collator = DataCollatorForTokenClassification(tokenizer)

        for param in model.parameters():
            param.requires_grad = True

        args = TrainingArguments(
            f"ner-19500-{fold}",
            evaluation_strategy = "epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=5,
            save_strategy='epoch',
            report_to='tensorboard',
            load_best_model_at_end=True,
            # gradient_accumulation_steps=2,
            # eval_accumulation_steps=2,
            # max_grad_norm=2,
            warmup_ratio=0.1,
            metric_for_best_model="f1"

        )

        trainer = Trainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["val"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        trainer.evaluate()
        
        predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
        
        if fold == 0:
            probs = predictions
        else:
            probs += predictions
            
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        print(results)
        print()
        del trainer
        model.to('cpu')
        gc.collect()
        # all_trainers.append(trainer)

    probs /= folds
    predictions = np.argmax(probs, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    print(results)
