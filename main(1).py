import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#   try:
#     # 只使用第一个GPU
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#   except RuntimeError as e:
#     # 有些设备不支持设置可见设备
#     print(e)


# 读取CSV文件并存储为pandas DataFrame
df = pd.read_csv('noorder.csv', header=None)

# 将第一列的特征和第二列的标签分别存储为X和y
data = (np.array([eval(x) for x in df.iloc[:, 0]])/255.0).astype(np.float32)
labels = np.array(df.iloc[:, 1]) - 1


train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2)

model = tf.keras.Sequential([
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Dense(128, activation='relu'),

  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=50, callbacks=[early_stop])

test_pred = model.predict(val_data)
test_pred = np.argmax(test_pred, axis=1)
f1 = f1_score(val_labels, test_pred, average='macro')
precision = precision_score(val_labels, test_pred, average='macro')
recall = recall_score(val_labels, test_pred, average='macro')
accuracy = accuracy_score(val_labels, test_pred)

print('F1 score: {:.2f}'.format(f1))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('Accuracy: {:.2f}'.format(accuracy))

# Plot the training history
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('Training history')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.legend()
plt.show()

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(data).batch(1).take(100):
    # Model has only one input so each data point has one element.
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(data).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32

tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

import pathlib

tflite_models_dir = pathlib.Path("")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Save the unquantized/float model:
tflite_model_file = tflite_models_dir/"model.tflite"
tflite_model_file.write_bytes(tflite_model)
# Save the quantized model:
tflite_model_quant_file = tflite_models_dir/"model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_model_quant)