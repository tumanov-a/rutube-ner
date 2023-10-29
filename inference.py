from transformers import AutoTokenizer, AutoModelForTokenClassification
from razdel import tokenize
import pandas as pd
import re
import torch
from transformers import pipeline
from tqdm import tqdm

model_checkpoints = ["ner-0/checkpoint-1446", "ner-1/checkpoint-1446"]
df = pd.read_csv("submission_example.csv").head(100)
df['entities_prediction'] = df.entities_prediction.apply(lambda x: x[2:-2].split("', '"))

tokenizer = AutoTokenizer.from_pretrained(model_checkpoints[0], device='cpu')
models = [AutoModelForTokenClassification.from_pretrained(model_checkpoint) for model_checkpoint in model_checkpoints]
id2label = models[0].config.id2label

submission = pd.DataFrame(columns=[['video_info', 'entities_prediction']])
submission['entities_prediction'] = submission['entities_prediction'].astype('object')


for i, elem in tqdm(df.iterrows()):
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
    submission.loc[i, 'entities_prediction'] = [[label] for label in labels]

submission.to_csv("submission.csv", index=False)