import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
m = tf.keras.models.load_model(r"D:\5. Academia\4. Machine Learning Engineering - FIAP\projeto_4\lstm_financial\models\etapa_B4_lstm_gru_best.keras")
l = m.layers[0]
print("CLASS:", l.__class__.__name__)
print("NAME:", l.name)
try:
    print("OUTPUT_SHAPE_ATTR:", l.output_shape)
except:
    print("OUTPUT_SHAPE_ATTR: NO ATTR")
try:
    print("OUTPUT_TENSOR_SHAPE:", l.output.shape)
except:
    print("OUTPUT_TENSOR_SHAPE: NO ATTR")
print("CONFIG:", l.get_config())
print("MODEL_INPUT_SHAPE:", m.input_shape)
