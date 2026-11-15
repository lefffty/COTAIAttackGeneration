import os

from tensorflow import keras


def build_model(input_dim: int) -> keras.Model:
    """Создаёт и компилирует MLP: Dense(64, relu) → Dense(32, relu) → Dense(2, softmax), Adam + SparseCategoricalCrossentropy."""
    m = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(2, activation="softmax"),
        ]
    )
    m.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return m


def train_or_load(X_train, y_train, X_test, y_test, path, epochs, batch_size):
    """Загружает модель с диска по path или обучает новую на train с валидацией на test и сохраняет в path."""
    if os.path.exists(path):
        return keras.models.load_model(path)
    m = build_model(X_train.shape[1])
    m.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    m.save(path)
    return m
