# Movie-recommender
This project build multiple movie recommendation system and then combine and compare them. Here is the file structure:
```
Movie-recommender/
├── README.md
├── evaluate_component
│   ├── __init__.py
│   └── evaluate.py
├── general_info
│   └── general_info.py
├── ml-latest-small
│   ├── README.txt
│   ├── links.csv
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── models
│   ├── __init__.py
│   ├── baseRecommender
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── abstract_recommender.py
│   ├── contentBased.py
│   ├── hybridExplicitFeedback.py
│   ├── hybridImplicitFeedback.py
│   ├── itemBasedCF.py
│   ├── popularity.py
│   └── userBasedCF.py
└── run_and_eval_all_models.ipynb
```

**Where:**
- `evaluate_component`: the model evaluating module
- `general_info`: the module containing general information in the dataset such as the genres of the movies which is fixed
- `ml-latest-small`: contains the [MovieLens dataset](https://grouplens.org/datasets/movielens/)
- `models`: contains several models to be combined by the `hybridExplicitFeedback` model and compared together
- `run_and_eval_all_models.ipynb`: this notebook test each model separately and some combination of them

More detailed information can be found in each directory
