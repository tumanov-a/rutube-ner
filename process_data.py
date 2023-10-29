import pandas as pd
import json
from razdel import tokenize

def extract_labels(item):
    
    # воспользуемся удобным токенайзером из библиотеки razdel, 
    # она помимо разбиения на слова, сохраняет важные для нас числа - начало и конец слова в токенах
    
    raw_toks = list(tokenize(item['video_info']))
    words = [tok.text for tok in raw_toks]
    # присвоим для начала каждому слову тег 'О' - тег, означающий отсутствие NER-а
    word_labels = ['O'] * len(raw_toks)
    char2word = [None] * len(item['video_info'])
    # так как NER можем состаять из нескольких слов, то нам нужно сохранить эту инфорцию
    for i, word in enumerate(raw_toks):
        char2word[word.start:word.stop] = [i] * len(word.text)

    labels = item['entities']
    if isinstance(labels, dict):
        labels = [labels]
    if labels is not None:
        for e in labels:
            if e['label'] != 'не найдено':
                e_words = sorted({idx for idx in char2word[e['offset']:e['offset']+e['length']] if idx is not None})
                if e_words:
                    word_labels[e_words[0]] = 'B-' + e['label']
                    for idx in e_words[1:]:
                        word_labels[idx] = 'I-' + e['label']
                else:
                    continue
            else:
                continue
        return {'tokens': words, 'tags': word_labels}
    else: return {'tokens': words, 'tags': word_labels}

def preprocess_data(filepath):
    data = pd.read_csv(filepath)  # "ner_data_train.csv"

    df = data.copy()
    df['entities'] = df['entities'].apply(lambda l: l.replace('\,', ',')if isinstance(l, str) else l)
    df['entities'] = df['entities'].apply(lambda l: l.replace('\\\\', '\\')if isinstance(l, str) else l)
    df['entities'] = df['entities'].apply(lambda l: '[' + l + ']'if isinstance(l, str) else l)
    df['entities'] = df['entities'].apply(lambda l: json.loads(l)if isinstance(l, str) else l)

    ner_data = [extract_labels(item) for i, item in df.iterrows()]

    return ner_data
