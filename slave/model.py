import numpy as np
import tflite_runtime.interpreter as tflite
from config import *
from mfcc import compute_manual_mfcc

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]


def predict(audio):
    mfcc = compute_manual_mfcc(audio)
    mfcc = (mfcc + 11.5) * (110 / 65) - 30

    input_scale, input_zero_point = input_details['quantization']
    output_scale, output_zero_point = output_details['quantization']

    mfcc_quantized = (mfcc / input_scale) + input_zero_point
    mfcc_quantized = np.clip(mfcc_quantized, -128, 127)
    mfcc_quantized = mfcc_quantized[np.newaxis, ..., np.newaxis].astype(np.int8)

    interpreter.set_tensor(input_details['index'], mfcc_quantized)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details['index'])
    prob = (output_data[0][0].astype(np.float32) - output_zero_point) * output_scale

    return prob