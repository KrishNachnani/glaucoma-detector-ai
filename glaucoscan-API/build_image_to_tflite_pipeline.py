import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import tensorflow as tf

# CPU only
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import numpy as np
import tensorflow as tf
import joblib
import pickle

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image as keras_image


SCALER_PATH = "models/scaler.joblib"
MLP_PATH = "models/glaucoma_mlp_model.pkl"
FEATURE_SELECTOR_PATH = "models/feature_selector.joblib"   
CLASS_MAPPING_PATH = "models/class_mapping.pkl"           
TFLITE_OUTPUT_PATH = "models/glaucoma_image_to_prediction.tflite"

IMG_SIZE = (224, 224)  # must match how you featurized originally


# Load pipeline objects
print("Loading scaler, MLP, and feature selector (if present)...")
scaler = joblib.load(SCALER_PATH)
with open(MLP_PATH, "rb") as f:
    mlp = pickle.load(f)

sfm = None
if FEATURE_SELECTOR_PATH is not None and os.path.exists(FEATURE_SELECTOR_PATH):
    sfm = joblib.load(FEATURE_SELECTOR_PATH)
    print("Feature selector loaded.")
else:
    print("No feature selector found / used.")

class_mapping = None
if CLASS_MAPPING_PATH is not None and os.path.exists(CLASS_MAPPING_PATH):
    with open(CLASS_MAPPING_PATH, "rb") as f:
        class_mapping = pickle.load(f)  # typically {"label": {0: "Glaucoma", 1: "No Glaucoma"}}
    print("Class mapping loaded:", class_mapping)
else:
    print("No class mapping file found.")


# Build Keras model for feature-level MLP
print("Building Keras MLP pipeline from scikit-learn objects...")

# Number of raw input features (before selection)
n_input_feats = scaler.mean_.shape[0]

# Keras input: raw feature vector
features_input = tf.keras.Input(shape=(n_input_feats,), name="raw_features")

#StandardScaler
mean = tf.constant(scaler.mean_, dtype=tf.float32, name="scaler_mean")
scale = tf.constant(scaler.scale_, dtype=tf.float32, name="scaler_scale")

def standardize(x):
    return (x - mean) / scale

x = tf.keras.layers.Lambda(standardize, name="standard_scaler")(features_input)

# Feature selection
if sfm is not None:
    selected_indices = sfm.get_support(indices=True)
    selected_indices_tf = tf.constant(selected_indices, dtype=tf.int32, name="selected_indices")

    def select_features(x_):
        return tf.gather(x_, selected_indices_tf, axis=1)

    x = tf.keras.layers.Lambda(select_features, name="feature_selection")(x)

# Recreate the MLPClassifier in Keras with same weights
coefs = mlp.coefs_           
intercepts = mlp.intercepts_ 

activation_map = {
    "identity": "linear",
    "logistic": "sigmoid",
    "tanh": "tanh",
    "relu": "relu",
}
hidden_activation = activation_map.get(mlp.activation, "relu")

keras_layers = []
# Hidden layers (all but last)
for layer_idx in range(len(coefs) - 1):
    n_units = intercepts[layer_idx].shape[0]
    dense = tf.keras.layers.Dense(
        n_units,
        activation=hidden_activation,
        name=f"hidden_{layer_idx}",
    )
    x = dense(x)
    keras_layers.append(dense)

# Output layer
n_out = intercepts[-1].shape[0]
out_act = getattr(mlp, "out_activation_", None)
if out_act == "logistic":
    output_activation = "sigmoid"   # binary classifier
elif out_act == "softmax":
    output_activation = "softmax"   # multi-class
else:
    output_activation = "linear"

output_dense = tf.keras.layers.Dense(
    n_out,
    activation=output_activation,
    name="output",
)
mlp_outputs = output_dense(x)
keras_layers.append(output_dense)

keras_mlp_pipeline = tf.keras.Model(
    inputs=features_input,
    outputs=mlp_outputs,
    name="glaucoma_mlp_pipeline_from_pkl"
)

# Set weights from scikit-learn MLP
for i, layer in enumerate(keras_layers):
    w = coefs[i].astype(np.float32)
    b = intercepts[i].astype(np.float32)
    layer.set_weights([w, b])

print("Feature-level MLP pipeline summary:")
keras_mlp_pipeline.summary()


# Prepend CNN backbone: image -> features -> MLP pipeline
print("Building full image-to-prediction model with ResNet50 backbone...")

# Backbone: must match how you originally computed features
base_cnn = ResNet50(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)

# Freeze backbone
base_cnn.trainable = False

# Image input
image_input = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

# Preprocess image as ResNet50 expects
x_img = preprocess_input(image_input)   
features = base_cnn(x_img)            

# Feed CNN features into MLP pipeline
predictions = keras_mlp_pipeline(features)

image_to_pred_model = tf.keras.Model(
    inputs=image_input,
    outputs=predictions,
    name="glaucoma_image_to_prediction_model"
)

print("Full image-to-prediction model summary:")
image_to_pred_model.summary()

# Convert full model to TFLite (image -> prediction)
print(f"Converting full image-to-prediction model to TFLite: {TFLITE_OUTPUT_PATH} ...")

converter = tf.lite.TFLiteConverter.from_keras_model(image_to_pred_model)
tflite_model = converter.convert()

with open(TFLITE_OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model to {TFLITE_OUTPUT_PATH}")
print("Done.")
