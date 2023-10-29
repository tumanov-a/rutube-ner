# NER Rutube 2023 Цифровой прорыв - команда Наносемантика

Данный сервис позволяет классифицировать сущности в описаниях и заголовках под видео Rutube.

Скачать модели можно по ссылке https://drive.google.com/drive/folders/1m_px60Rd1UPYEfuCvCvBDNtxg0cii3dn?usp=drive_link и загрузить в папку models.
`Checkpoint-12500` - это претрейненный MLM чекпоинт `xlm-roberta-large`.
Другие 3 архива - это модели с разных фолдов, которые в совокупности дали наилучшее качество на сабмишене.

 - process_data.py - Скрипт для препроцессинга данных
 - train-kfold.py, train-kfold2.py, train-kfold3.py - Обучение модели
 - inference.py - Скрипт инференса для расчета submission
 - visualization.ipynb - Ноутбук с визуализацией предсказаний модели
