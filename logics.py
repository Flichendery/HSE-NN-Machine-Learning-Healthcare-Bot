import json
import pandas as pd

def load_diseases_dict(file_path):
    """
    Загружает словарь болезней из JSON-файла.
    
    Args:
        file_path (str): Путь к JSON-файлу
    
    Returns:
        dict: Словарь с кодами МКБ-10 в качестве ключей и названиями болезней в качестве значений
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            diseases = json.load(file)
        return diseases
    except FileNotFoundError:
        print(f"Файл {file_path} не найден.")
        return {}
    except json.JSONDecodeError:
        print(f"Ошибка при чтении JSON-файла {file_path}.")
        return {}


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return pd.DataFrame(items)


def load_mkb_codes(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    int_data = {int(k): v for k, v in data.items()}
    return int_data

    