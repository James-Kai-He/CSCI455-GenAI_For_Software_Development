### Setup

Open the folder A1. Create a python enviroment and activate it:

```
python3 -m venv venv
```

```
source venv/bin/activate
```

Install dependencies:
```
pip3 install -r requirements.txt
```

### Data Pipeline

Run data_preparation.py to generate the dataset.

```
python3 data_preparation.py
```

Data will be outputted into ./datasets/java_repos and ./datasets/ngram_dataset

### Training and evaluating N-Grams

Then run ngram.py to train the ngrams with hyperparameter tuning and generate the json output files.

The parameters are n = [3,5,7] and alpha = [0.001, 0.01, 0.1, 0.5, 1.0]

```
python3 ngram.py
```

The output files will be located inside ./datasets/ngram_dataset.