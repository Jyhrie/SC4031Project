import numpy as np
from config import *
from mfcc import compute_mfcc

try:
    import tflite_runtime.interpreter as tflite
    tflite_interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    tflite_interpreter = tf.lite.Interpreter

interpreter = tflite_interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


def predict(audio):
    mfcc = compute_mfcc(audio)

    mfcc = (mfcc + 11.5) * (110 / 65) - 30

    mfcc = mfcc[np.newaxis, ..., np.newaxis].astype(np.int8)

    interpreter.set_tensor(input_details["index"], mfcc)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details["index"])
    return float(output[0][0])