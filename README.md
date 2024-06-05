# Инструкция по запуску сервисов Preprocessing и ML

## Предварительные шаги

1. Убедитесь, что у вас установлен Python 3.7 или выше.
2. Установите необходимые библиотеки с помощью pip:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. Убедитесь, что у вас есть файл модели resnet101.pth в папке, установвленной в файле config.py.

## Запуск сервисов

    uvicorn main:app --port 8000 --workers num_workers

**num_workers** - количество одновременно запущенных процессов.