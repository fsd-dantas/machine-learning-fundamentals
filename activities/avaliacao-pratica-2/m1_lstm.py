"""
LSTM classifier — the recurrent arm of the comparison.

Follows the lecture notebook's architecture (Embedding -> 2 stacked LSTM layers with
dropout -> Dense -> softmax) with four corrections, each of which changes the number
that gets reported:

1. **`max_length` 20 -> 48.** The corpus has a *median* of 30 words and a 95th
   percentile of 44. Truncating at 20 discards a third of every headline before the
   model sees it — including, routinely, the clause that carries the sentiment.

2. **Early stopping on a real validation split.** The notebook trains a fixed 20 epochs
   and passes its 30% holdout as `validation_data`, then reports accuracy on that same
   holdout. That is model selection on the test set. Here the validation slice comes out
   of the training 70% (see `common.make_splits`) and the test set is read once.

3. **The tokeniser is fitted on training text only.** The notebook already does this
   correctly. It is worth stating because it is the step everyone gets wrong: fitting a
   vocabulary on the full corpus leaks test-set word statistics into training.

4. **Seeded and repeated.** The notebook fixes no seed, so its accuracy is not
   reproducible even by itself.

Stopword removal is kept, faithful to the lecture, and is a defensible choice *for an
LSTM with randomly-initialised embeddings*: with ~1,450 training texts there is not
enough signal to learn that "de" and "a" are uninformative, so removing them by hand
spends the tiny capacity on words that matter. It is the wrong choice for BERT, and
`m2_bert.py` explains why.

Usage:
    python m1_lstm.py --task binary
    python m1_lstm.py --task multiclass --bidirectional --seed 7
"""

from __future__ import annotations

import argparse
import re

import numpy as np

import common

VOCAB_SIZE = 5000
EMBEDDING_DIM = 128
MAX_LENGTH = 48        # p95 of the corpus is 44 words; the lecture used 20
OOV_TOKEN = "<NKN>"


def portuguese_stopwords() -> set[str]:
    from string import punctuation
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            words = stopwords.words("portuguese")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            words = stopwords.words("portuguese")
    except ImportError:
        words = []
    return set(words) | set(punctuation)


def strip_stopwords(texts: np.ndarray, stops: set[str]) -> np.ndarray:
    """Token-wise removal.

    The lecture does this with repeated `str.replace(' ' + word + ' ', ' ')`, which is
    O(vocabulary) passes per text *and* silently misses stopwords at the start or end of
    a text, since neither is surrounded by two spaces. Tokenising once and filtering is
    both faster and correct.
    """
    out = []
    for text in texts:
        tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        out.append(" ".join(t for t in tokens if t not in stops))
    return np.array(out)


def build_model(n_classes: int, seed: int, bidirectional: bool):
    import tensorflow as tf
    from tensorflow import keras

    layers = keras.layers
    init = keras.initializers.GlorotUniform(seed=seed)

    def recurrent(units, return_sequences):
        cell = layers.LSTM(units, dropout=0.25, return_sequences=return_sequences,
                           kernel_initializer=init)
        return layers.Bidirectional(cell) if bidirectional else cell

    model = keras.Sequential([
        keras.Input(shape=(MAX_LENGTH,)),
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True),
        recurrent(EMBEDDING_DIM, True),
        recurrent(EMBEDDING_DIM, False),
        layers.Dense(64, activation="relu", kernel_initializer=init),
        layers.Dropout(0.3, seed=seed),
        layers.Dense(n_classes, activation="softmax", dtype="float32",
                     kernel_initializer=init),
    ], name="lstm")
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.Adam(1e-3), metrics=["accuracy"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="binary", choices=common.TASKS)
    parser.add_argument("--seed", type=int, default=common.SEED)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--keep-stopwords", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    common.set_seed(args.seed)
    corpus = common.load_corpus(args.task)
    splits = common.make_splits(corpus, args.seed)
    device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    print(f"device: {device} · {splits.summary()}\n")

    x_train, x_val, x_test = splits.x_train, splits.x_val, splits.x_test
    if not args.keep_stopwords:
        stops = portuguese_stopwords()
        x_train = strip_stopwords(x_train, stops)
        x_val = strip_stopwords(x_val, stops)
        x_test = strip_stopwords(x_test, stops)

    # Fitted on training text alone: the vocabulary must not know what is in the test set.
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(x_train)

    def encode(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LENGTH,
                             padding="post", truncating="post")

    train, val, test = encode(x_train), encode(x_val), encode(x_test)
    model = build_model(corpus.n_classes, args.seed, args.bidirectional)
    model.summary()

    early = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6,
                                          mode="max", restore_best_weights=True)
    with common.Timer() as timer:
        history = model.fit(train, splits.y_train,
                            validation_data=(val, splits.y_val),
                            epochs=args.epochs, batch_size=args.batch_size,
                            callbacks=[early], verbose=2)

    y_pred = np.argmax(model.predict(test, verbose=0), axis=1)
    name = "bilstm" if args.bidirectional else "lstm"

    common.score_run(
        task=args.task, model=name, seed=args.seed,
        config={"vocab_size": VOCAB_SIZE, "embedding_dim": EMBEDDING_DIM,
                "max_length": MAX_LENGTH, "bidirectional": args.bidirectional,
                "stopwords_removed": not args.keep_stopwords,
                "embeddings": "learned from scratch", "optimizer": "Adam(1e-3)",
                "vocabulary_seen": len(tokenizer.word_index)},
        corpus=corpus, y_true=splits.y_test, y_pred=y_pred,
        params_trainable=sum(int(np.prod(w.shape)) for w in model.trainable_weights),
        epochs_run=len(history.history["val_accuracy"]),
        train_seconds=timer.seconds, device=device,
    )


if __name__ == "__main__":
    main()
