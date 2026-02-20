import os
import json
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm


DATASET_DIR = "./datasets/ngram_dataset"

TRAIN_FILES = {
    "T1": os.path.join(DATASET_DIR, "train_t1.txt"),
    "T2": os.path.join(DATASET_DIR, "train_t2.txt"),
    "T3": os.path.join(DATASET_DIR, "train_t3.txt"),
}
VAL_FILE           = os.path.join(DATASET_DIR, "val.txt")
PROVIDED_TEST_FILE = os.path.join(DATASET_DIR, "provided_test.txt")
SELF_TEST_FILE     = os.path.join(DATASET_DIR, "test.txt")

N_VALUES     = [3, 5, 7]
ALPHA_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0]
ALPHA        = 0.1
CUTOFF_FREQ  = 1


def load_dataset(filepath):
    methods = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                methods.append(line.split())
    return methods



class NgramLM(object):
    def __init__(self, n, sos="<s>", eos="</s>"):
        # n: size of n-gram (e.g., 3 for trigram)
        # sos: start-of-sequence token
        # eos: end-of-sequence token
        # unk: unknown token for out-of-vocabulary words

        self.n   = n
        self.sos = sos
        self.eos = eos
        self.unk = "[UNK]"

        if self.n < 2:
            raise Exception("Size of n-gram must be at least 2")

        self.ngram_counts = defaultdict(int)

        self.vocabulary      = set()
        self.vocabulary_size = 0

        self._context_index = None


    def add_padding(self, token_lists):
        # Pad each token list with (n-1) SOS tokens at the start and (n-1) EOS tokens at the end
        padded_token_lists = []

        for token_list in token_lists:
            padded_token_lists.append(
                [self.sos] * (self.n - 1) + token_list + [self.eos] * (self.n - 1)
            )

        return padded_token_lists


    def vocab_lookup(self, sequence):
        # Map tokens not in the vocabulary to [UNK]
        output = None
        if isinstance(sequence, str):
            output = " ".join(
                [w.strip() if w.strip() in self.vocabulary else self.unk
                 for w in sequence.split()]
            ).strip()
        elif isinstance(sequence, list):
            output = [
                w.strip() if w.strip() in self.vocabulary else self.unk
                for w in sequence
            ]
        return output


    def build_vocabulary(self, texts, cutoff_freq):
        # Build vocabulary from the training data, including only words that appear at least cutoff_freq times
        vocabulary  = set()
        word_counts = {}

        for text in texts:
            for word in text:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, freq in word_counts.items():
            if freq >= cutoff_freq:
                vocabulary.add(word)

        self.vocabulary      = vocabulary
        self.vocabulary_size = len(self.vocabulary)


    def generate_ngrams(self, token_list):
        # Generate n-grams and (n-1)-grams from the token list
        ngrams = []

        for i in range(len(token_list) - self.n + 1):
            ngram = tuple(token_list[i : i + self.n])
            ngrams.append(ngram)

        for i in range(len(token_list) - (self.n - 1) + 1):
            context_ngram = tuple(token_list[i : i + (self.n - 1)])
            ngrams.append(context_ngram)

        return ngrams


    def fit(self, padded_token_lists):
        # Count n-grams and (n-1)-grams in the training data
        self.ngram_counts = defaultdict(int)

        for token_list in tqdm(padded_token_lists):
            token_list = self.vocab_lookup(token_list)
            ngrams = self.generate_ngrams(token_list)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1

        # Build context index for fast argmax prediction
        self._build_context_index()

        return self


    def _build_context_index(self):
        index = defaultdict(dict)
        for ngram, count in self.ngram_counts.items():
            if len(ngram) == self.n:          # only full n-grams, not (n-1)-grams
                context   = ngram[:-1]
                next_tok  = ngram[-1]
                index[context][next_tok] = count
        self._context_index = index


    def calc_ngram_prob(self, ngram, k=0):
        # Calculate the probability of an n-gram using add-k smoothing
        context = ngram[:-1]

        count_ngram         = self.ngram_counts[ngram]
        count_context_ngram = self.ngram_counts[context]

        probability = (count_ngram + k) / (count_context_ngram + k * self.vocabulary_size)

        return probability


    def calc_sentence_logprobs(self, token_lists, k=0):
        # Calculate the log-probability of each sentence under the model
        padded_token_lists = self.add_padding(token_lists)
        ngram_lists = [self.generate_ngrams(t) for t in padded_token_lists]
        return [
            np.sum([np.log(self.calc_ngram_prob(ngram, k=k)) for ngram in ngram_list])
            for ngram_list in ngram_lists
        ]


    def calc_perplexity(self, token_lists, k=ALPHA):
        # Calculate perplexity of the model on the given token lists
        total_logp   = 0.0
        total_tokens = 0

        padded = self.add_padding(token_lists)

        for tokens in padded:
            mapped = self.vocab_lookup(tokens)
            for i in range(self.n - 1, len(mapped)):
                ngram = tuple(mapped[i - (self.n - 1) : i + 1])
                prob  = self.calc_ngram_prob(ngram, k=k)
                total_logp   += math.log(prob)
                total_tokens += 1

        return math.exp(-total_logp / total_tokens)


    def predict_next_token(self, context_tokens, k=ALPHA):
        context = tuple(context_tokens[-(self.n - 1):])

        candidates = self._context_index.get(context, None)

        if candidates:
            best_tok  = max(candidates, key=lambda tok: candidates[tok])
            best_prob = self.calc_ngram_prob(context + (best_tok,), k=k)
        else:
            best_tok  = self.unk
            best_prob = self.calc_ngram_prob(context + (self.unk,), k=k)

        return best_tok, best_prob



def train_model(train_tokens, n, cutoff_freq=CUTOFF_FREQ):
    # Train an n-gram language model on the given training tokens with specified cutoff frequency for vocabulary
    model  = NgramLM(n=n)
    padded = model.add_padding(train_tokens)
    model.build_vocabulary(padded, cutoff_freq=cutoff_freq)
    model.fit(padded)
    return model



def write_json(model, test_tokens_raw, set_label, perplexity, filepath, k=ALPHA):
    padded_seqs = model.add_padding(test_tokens_raw)
    total       = len(test_tokens_raw)

    with open(filepath, "w", encoding="utf-8") as f:

        f.write("{\n")
        f.write(f'  "testSet": {json.dumps(set_label)},\n')
        f.write(f'  "perplexity": {round(perplexity, 4)},\n')
        f.write('  "data": [\n')
        f.flush()

        for idx, (raw_tokens, padded_tokens) in enumerate(tqdm(
            zip(test_tokens_raw, padded_seqs),
            total=total,
            desc=f"Writing JSON ({set_label})",
        )):
            mapped_tokens  = model.vocab_lookup(padded_tokens)
            tokenized_code = " ".join(raw_tokens)
            predictions    = []

            for i in range(model.n - 1, len(mapped_tokens)):
                context  = list(mapped_tokens[i - (model.n - 1) : i])
                gt_token = mapped_tokens[i]

                pred_token, pred_prob = model.predict_next_token(context, k=k)

                predictions.append({
                    "context":         context,
                    "predToken":       pred_token,
                    "predProbability": round(pred_prob, 8),
                    "groundTruth":     gt_token,
                })

            method_entry = {
                "index":         f"ID{idx + 1}",
                "tokenizedCode": tokenized_code,
                "contextWindow": model.n,
                "predictions":   predictions,
            }

            entry_str = json.dumps(method_entry, indent=2)
            entry_str = "\n".join("    " + line for line in entry_str.splitlines())

            if idx < total - 1:
                f.write(entry_str + ",\n")
            else:
                f.write(entry_str + "\n")

            f.flush()
            predictions = None

        f.write("  ]\n")
        f.write("}\n")

    print(f"  Saved: {filepath}")



def main():
    print("Loading datasets...")

    train_data = {}
    for label, path in TRAIN_FILES.items():
        train_data[label] = load_dataset(path)

    val_data      = load_dataset(VAL_FILE)
    provided_test = load_dataset(PROVIDED_TEST_FILE)
    self_test     = load_dataset(SELF_TEST_FILE)



    print("\n")
    print(f"Training models  n = {N_VALUES},  sets {{T1, T2, T3}}")

    all_models = {} 

    for train_label, train_tokens in train_data.items():
        for n in N_VALUES:
            print(f"\n  Training {n}-gram model on {train_label}...")
            model = train_model(train_tokens, n, cutoff_freq=CUTOFF_FREQ)
            all_models[(train_label, n)] = model

            print(f"  Vocabulary size    : {model.vocabulary_size:,}")
            print(f"  Unique n-gram types: {len(model.ngram_counts):,}")


    print("\n")
    print(f"Alpha tuning  k = {ALPHA_VALUES}  on validation set")

    val_results = {}

    for (train_label, n), model in all_models.items():
        for alpha in ALPHA_VALUES:
            val_pp = model.calc_perplexity(val_data, k=alpha)
            val_results[(train_label, n, alpha)] = val_pp


    print("\n")
    print("Validation Perplexity:")
    alpha_header = "".join(f"{a:>10}" for a in ALPHA_VALUES)
    print(f"{'Config':<14}{alpha_header}")
    print("-" * (14 + 10 * len(ALPHA_VALUES)))
    for tl in ["T1", "T2", "T3"]:
        for n in N_VALUES:
            row = f"{tl + ' n=' + str(n):<14}"
            for alpha in ALPHA_VALUES:
                row += f"{val_results[(tl, n, alpha)]:>10.4f}"
            print(row)


    best_key                         = min(val_results, key=val_results.get)
    best_train_label, best_n, best_k = best_key
    best_val_pp                      = val_results[best_key]
    best_model                       = all_models[(best_train_label, best_n)]

    print(f"\nBest config: n={best_n}, training set={best_train_label}, k={best_k}")


    print("\n")
    print("Best model:")

    provided_pp = best_model.calc_perplexity(provided_test, k=best_k)
    self_pp     = best_model.calc_perplexity(self_test,     k=best_k)

    print(f"  Provided test perplexity    : {provided_pp:.4f}")
    print(f"  Self-created test perplexity: {self_pp:.4f}")


    print("\n")
    print("Generating JSON output files...")

    os.makedirs(DATASET_DIR, exist_ok=True)

    write_json(
        best_model,
        provided_test,
        set_label="provided.txt",
        perplexity=provided_pp,
        filepath=os.path.join(DATASET_DIR, "results-provided.json"),
        k=best_k,
    )
    write_json(
        best_model,
        self_test,
        set_label="test.txt",
        perplexity=self_pp,
        filepath=os.path.join(DATASET_DIR, "results-test.json"),
        k=best_k,
    )

    print("\nOutput files:")
    print(f"  {DATASET_DIR}/results-provided.json")
    print(f"  {DATASET_DIR}/results-selfcreated.json")


if __name__ == "__main__":
    main()