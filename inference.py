from transformers import AutoTokenizer, AutoModelForTokenClassification
from razdel import tokenize
import pandas as pd
import re
import torch
from transformers import pipeline
from tqdm import tqdm


if __name__ == '__main__':
    # model_checkpoints = ["ner-19500-2/checkpoint-1928", "ner-v2-1/checkpoint-1716", "ner-19500-0/checkpoint-2410", 'checkpoint-1446']
    # model_checkpoints = ["ner-v2-0/checkpoint-1712", "ner-v2-1/checkpoint-1716", "ner-v2-2/checkpoint-1716"]
    model_checkpoints = ['ner-v3-0/checkpoint-1284', 'ner-v3-1/checkpoint-1287']
    # submission = pd.read_csv("data/ner_data_test.csv").head(10)
    # print(df.isnull().sum())
    # df['entities_prediction'] = df.entities_prediction.apply(lambda x: x[2:-2].split("', '"))

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoints[0], device='cpu')
    models = [AutoModelForTokenClassification.from_pretrained(model_checkpoint).to('cuda') for model_checkpoint in model_checkpoints]
    id2label = models[0].config.id2label

    submission = pd.read_csv("data/ner_data_test.csv")
    submission['entities_prediction'] = submission['entities_prediction'].astype('object')

    for i, elem in tqdm(submission.iterrows()):
        razdel_tokens = tokenize(elem['video_info'])
        razdel_tokens = [tok.text for tok in razdel_tokens]
        
        dismissed_token = re.compile(r'\xad+|\u200b+')
        razdel_tokens = [re.sub(dismissed_token, '[UNK]', tok) for tok in razdel_tokens]
        
        tokens = tokenizer(razdel_tokens, truncation=True, is_split_into_words=True, return_tensors='pt')
        words = tokens.words()

        tokens = {k: v.to(models[0].device) for k, v in tokens.items()}

        with torch.no_grad():
            for n_fold, model in enumerate(models):
                if n_fold == 0:
                    logits = model(**tokens).logits
                else:
                    logits += model(**tokens).logits
        logits /= len(models)
        indices = logits.argmax(dim=-1)[0].cpu().numpy()
        labels = []
        prev=words[1] # это всегда ноль - первое слово
        labels = [id2label[indices[1]]]
        for word, tag in zip(words[1:-1], indices[1:-1]):
            if word != prev:
                labels.append(id2label[tag])
                prev=word
        
        if len(razdel_tokens)!= len(labels):
                print(tokens['tokens'])
        submission.loc[i, 'video_info'] = elem['video_info']
        submission.loc[i, 'entities_prediction'] = [label for label in labels]

    submission.to_csv("submissions/submission_ner-v3-0_checkpoint-1284_ner-v3-1_checkpoint-1287.csv", index=False)
