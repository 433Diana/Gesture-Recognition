import tensorflow as tf

tflite_model= tf.lite.Interpreter(model_path='model_quan.tflite')  # .contrib
tflite_model.allocate_tensors()
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()

print(input_details)
print(output_details)

