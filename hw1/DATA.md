# Рекомендательные системы на KuaiRec
ФИО: Хомяков Максимилиан Валерьевич


## Модели

Реализации находятся в `models/`:

- **ItemCF** - рекомендации по сходству видео.
- **iALS** - матричная факторизация для неявной обратной связи.
- **EASE** - линейная модель с регуляризацией.
- **FM** - факторизационная машина с ID пользователей, ID видео и категориями.
- **CustomALS** - iALS с дополнительным целевым сигналом слабых просмотров.
- **Popularity** - базовая модель по популярности видео.

## Результаты

- [experiments.ipynb](experiments.ipynb) - анализ данных, подбор параметров и сравнение моделей.
- [results/final_test.csv](results/final_test.csv) - итоговая таблица с гиперпараметрами, Recall@10 и NDCG@10.
- [results/test_ndcg10.png](results/test_ndcg10.png) и [results/test_recall10.png](results/test_recall10.png) - графики сравнения.
- [results/fm_validation.csv](results/fm_validation.csv) - результаты подбора параметров FM.

## Датасет

Используется [KuaiRec 2.0](https://github.com/chongminggao/KuaiRec).
Архив доступен на [Zenodo](https://zenodo.org/records/18164998/files/KuaiRec.zip).
Нужны файлы `big_matrix.csv` и `item_categories.csv`.

Загрузка из директории `hw1/`:

```bash
mkdir -p data/raw
curl --fail --location --retry 5 \
  https://zenodo.org/records/18164998/files/KuaiRec.zip \
  --output data/KuaiRec.zip
unzip -q data/KuaiRec.zip -d data/raw
```

## Запуск

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r hw1/requirements.txt
cd hw1
python -m jupyterlab experiments.ipynb
```

Выберите ядро окружения и выполните **Restart Kernel and Run All Cells**.
Таблицы и графики сохраняются в `results/`, выводы находятся в конце ноутбука.
