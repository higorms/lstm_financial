"""
Construção das Arquiteturas LSTM.

Conforme PLANEJAMENTO.md - Seções 4.1 e 4.2:
- Arquitetura Base (baseline)
- B1: Attention LSTM
- B2: Conv1D + LSTM
- B3: Bidirectional LSTM
- B4: LSTM + GRU Híbrido
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model, regularizers


def _l2(val: float):
    """Helper: retorna regularizer L2 ou None se val=0."""
    return regularizers.l2(val) if val > 0 else None


def build_baseline(
    input_shape: tuple,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 32,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.2,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
) -> Model:
    """
    Arquitetura Base — Seção 4.1.

    Input → LSTM(128) → BN → Drop → LSTM(64) → BN → Drop → Dense(32) → Dense(1, sigmoid)
    L2 regularization on kernel + recurrent weights.
    """
    reg = _l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.LSTM(lstm1_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_1")(inputs)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=False,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_2")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=reg, name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="baseline_lstm")
    return model


def build_attention_lstm(
    input_shape: tuple,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 32,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.2,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
) -> Model:
    """
    B1: Attention LSTM — Seção 4.2.

    LSTM(return_seq=True) → LSTM(return_seq=True) → Self-Attention → GlobalAvgPool → Dense
    """
    reg = _l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.LSTM(lstm1_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_1")(inputs)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_2")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    # Self-Attention
    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=lstm2_units // 4, name="attention"
    )(x, x)
    x = layers.Add(name="residual")([x, attention])
    x = layers.LayerNormalization(name="layer_norm")(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=reg, name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="attention_lstm")
    return model


def build_conv1d_lstm(
    input_shape: tuple,
    conv_filters: int = 64,
    conv_kernel: int = 3,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 32,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.2,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
) -> Model:
    """
    B2: Conv1D + LSTM — Seção 4.2.

    Conv1D → MaxPool → BN → LSTM → LSTM → Dense
    """
    reg = _l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Conv1D(conv_filters, conv_kernel, activation="relu",
                       padding="same", kernel_regularizer=reg,
                       name="conv1d")(inputs)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_conv")(x)

    x = layers.LSTM(lstm1_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_1")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    x = layers.LSTM(lstm2_units, return_sequences=False,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_2")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=reg, name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="conv1d_lstm")
    return model


def build_bidirectional_lstm(
    input_shape: tuple,
    lstm1_units: int = 128,
    lstm2_units: int = 64,
    dense_units: int = 32,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.2,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
) -> Model:
    """
    B3: Bidirectional LSTM — Seção 4.2.

    Bidirectional(LSTM) → Bidirectional(LSTM) → Dense
    """
    reg = _l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.Bidirectional(
        layers.LSTM(lstm1_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_1"
    )(inputs)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    x = layers.Bidirectional(
        layers.LSTM(lstm2_units, return_sequences=False,
                     kernel_regularizer=reg, recurrent_regularizer=reg),
        name="bilstm_2"
    )(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=reg, name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="bidirectional_lstm")
    return model


def build_lstm_gru_hybrid(
    input_shape: tuple,
    lstm_units: int = 128,
    gru_units: int = 64,
    dense_units: int = 32,
    dropout_lstm: float = 0.3,
    dropout_dense: float = 0.2,
    use_batchnorm: bool = True,
    l2_reg: float = 1e-4,
) -> Model:
    """
    B4: LSTM + GRU Híbrido — Seção 4.2.

    LSTM(128) → GRU(64) → Dense
    """
    reg = _l2(l2_reg)
    inputs = layers.Input(shape=input_shape, name="input")

    x = layers.LSTM(lstm_units, return_sequences=True,
                     kernel_regularizer=reg, recurrent_regularizer=reg,
                     name="lstm_1")(inputs)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_1")(x)

    x = layers.GRU(gru_units, return_sequences=False,
                    kernel_regularizer=reg, recurrent_regularizer=reg,
                    name="gru_1")(x)
    if use_batchnorm:
        x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_lstm, name="dropout_2")(x)

    x = layers.Dense(dense_units, activation="relu",
                      kernel_regularizer=reg, name="dense_1")(x)
    x = layers.Dropout(dropout_dense, name="dropout_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lstm_gru_hybrid")
    return model


# Registro de todas as arquiteturas disponíveis
ARCHITECTURES = {
    "baseline": build_baseline,
    "attention": build_attention_lstm,
    "conv1d_lstm": build_conv1d_lstm,
    "bidirectional": build_bidirectional_lstm,
    "lstm_gru": build_lstm_gru_hybrid,
}


def build_model(
    architecture: str,
    input_shape: tuple,
    **kwargs,
) -> Model:
    """
    Factory para construir qualquer arquitetura pelo nome.

    Parameters
    ----------
    architecture : str
        Nome da arquitetura: 'baseline', 'attention', 'conv1d_lstm',
        'bidirectional', 'lstm_gru'.
    input_shape : tuple
        Shape de entrada (timesteps, features).
    **kwargs
        Hiperparâmetros da arquitetura.
    """
    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Arquitetura '{architecture}' não encontrada. "
            f"Disponíveis: {list(ARCHITECTURES.keys())}"
        )
    return ARCHITECTURES[architecture](input_shape=input_shape, **kwargs)
