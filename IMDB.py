import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, SimpleRNN, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -----------------------------
# 1. Hyperparameters
# -----------------------------
vocab_size = 10000      # only keep top 10k most frequent words
maxlen = 200            # pad / cut each review to 200 tokens
embedding_dim = 64      # size of word embedding vectors
batch_size = 128
epochs = 5              # start small; increase later if needed

# -----------------------------
# 2. Load and preprocess IMDB dataset
# -----------------------------
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

# Pad sequences so all reviews have length = maxlen
x_train_padded = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test_padded  = pad_sequences(x_test,  maxlen=maxlen, padding='post', truncating='post')

# Create validation set from training data (with shuffling)
x_train_final, x_val, y_train_final, y_val = train_test_split(
    x_train_padded,
    y_train,
    test_size=0.2,
    shuffle=True,
    random_state=42
)

# -----------------------------
# 3. Model builders
# -----------------------------
def build_RNN_model(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        SimpleRNN(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model

def build_LSTM_model(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model

def build_GRU_model(vocab_size, embedding_dim, maxlen):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen),
        GRU(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model

# -----------------------------
# 4. Build models
# -----------------------------
rnn_model  = build_RNN_model(vocab_size, embedding_dim, maxlen)
lstm_model = build_LSTM_model(vocab_size, embedding_dim, maxlen)
gru_model  = build_GRU_model(vocab_size, embedding_dim, maxlen)

print("RNN model summary:")
rnn_model.summary()

print("\nLSTM model summary:")
lstm_model.summary()

print("\nGRU model summary:")
gru_model.summary()

# -----------------------------
# 5. Train models
# -----------------------------
history_rnn = rnn_model.fit(
    x_train_final, y_train_final,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1
)

history_lstm = lstm_model.fit(
    x_train_final, y_train_final,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1
)

history_gru = gru_model.fit(
    x_train_final, y_train_final,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    verbose=1
)

# -----------------------------
# 6. Plot training curves
# -----------------------------
def plot_history(histories, metric, title):
    """
    histories: dict like {'RNN': history_rnn, 'LSTM': history_lstm, 'GRU': history_gru}
    metric: 'loss' or 'accuracy'
    """
    plt.figure(figsize=(8, 5))
    
    for name, history in histories.items():
        values = history.history[metric]
        val_values = history.history[f'val_{metric}']
        epochs_range = range(1, len(values) + 1)
        
        plt.plot(epochs_range, values, marker='o', label=f'{name} train {metric}')
        plt.plot(epochs_range, val_values, marker='x', linestyle='--', label=f'{name} val {metric}')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()

histories = {
    'RNN': history_rnn,
    'LSTM': history_lstm,
    'GRU': history_gru
}

plot_history(histories, 'loss', 'Training vs Validation Loss')
plot_history(histories, 'accuracy', 'Training vs Validation Accuracy')

# -----------------------------
# 7. Evaluate on test set
# -----------------------------
test_loss_rnn,  test_acc_rnn  = rnn_model.evaluate(x_test_padded, y_test, verbose=0)
test_loss_lstm, test_acc_lstm = lstm_model.evaluate(x_test_padded, y_test, verbose=0)
test_loss_gru,  test_acc_gru  = gru_model.evaluate(x_test_padded, y_test, verbose=0)

print(f"Test RNN  - Loss: {test_loss_rnn:.4f}, Accuracy: {test_acc_rnn:.4f}")
print(f"Test LSTM - Loss: {test_loss_lstm:.4f}, Accuracy: {test_acc_lstm:.4f}")
print(f"Test GRU  - Loss: {test_loss_gru:.4f}, Accuracy: {test_acc_gru:.4f}")
