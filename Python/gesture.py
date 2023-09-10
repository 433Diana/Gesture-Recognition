import tempfile
import os

import tensorflow as tf
from tensorflow import keras

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np

path = './imgfolder'
nb_classes 		= 5
img_channels 	= 1
img_rows 		= 40
img_cols 		= 40
nbsize 			= 32
nepoch			= 20

def modlistdir(path, pattern = None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)
            
    return retlist

def initializers():
    imlist = modlistdir(path)
    
    image1 = np.array(Image.open(path +'/' + imlist[0])) # open one image to get size
    #image1 = np.resize(image1,(img_rows, img_cols))
    #plt.imshow(im1)
    
    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    
    print(immatrix.shape)
    
    #########################################################
    ## Label the set of images per respective gesture type.
    ##
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''
    
    data,Label = shuffle(immatrix,label, random_state=2) #Shuffle arrays or sparse matrices in a consistent way
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_channels)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_channels)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices (one_hot encoding)
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test

# Load MNIST dataset
X_train, X_test, Y_train, Y_test = initializers()

# Define the model architecture.
model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(40, 40, 1)),
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(5,activation='softmax')
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

model.fit(
    X_train,
    Y_train,
    batch_size=nbsize,
    epochs=nepoch,
    verbose=1,
    validation_split=0.1,
)

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(X_train, 
                  Y_train,
                  batch_size=nbsize, 
              	  epochs=nepoch, 
            	  verbose=1, 
                  validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(
    X_test, Y_test, verbose=0)

_, q_aware_model_accuracy = q_aware_model.evaluate(
   X_test, Y_test, verbose=0)

print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)

converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

quantized_tflite_model = converter.convert()

import numpy as np

def evaluate_model(interpreter):
	input_index = interpreter.get_input_details()[0]["index"]
	output_index = interpreter.get_output_details()[0]["index"]

	# Run predictions on every image in the "test" dataset.
	prediction_digits = []
	for i, test_image in enumerate(X_test):
		if i % 100 == 0:
			print('Evaluated on {n} results so far.'.format(n=i))
		# Pre-processing: add batch dimension and convert to float32 to match with
		# the model's input data format.
		test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
		interpreter.set_tensor(input_index, test_image)
		# Run inference.
		interpreter.invoke()

		# Post-processing: remove batch dimension and find the digit with highest
		# probability.
		output = interpreter.tensor(output_index)
		digit = np.argmax(output()[0])
		prediction_digits.append(digit)

	print('\n')
	# Compare prediction results with ground truth labels to calculate accuracy.
	prediction_digits = np.array(prediction_digits)
	Y_lable = np.array(Y_test).T
	Y_lable = tf.argmax(Y_lable)
	Y_lable = np.array(Y_lable)

	accuracy = (prediction_digits == Y_lable).mean()
	return accuracy

interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
interpreter.allocate_tensors()

test_accuracy = evaluate_model(interpreter)

print('Quant TFLite test_accuracy:', test_accuracy)
print('Quant TF test accuracy:', q_aware_model_accuracy)

# Create float TFLite model.
float_converter = tf.lite.TFLiteConverter.from_keras_model(model)
float_tflite_model = float_converter.convert()

# Measure sizes of models.
_, float_file = tempfile.mkstemp('.tflite')
_, quant_file = tempfile.mkstemp('.tflite')

with open(quant_file, 'wb') as f:
  f.write(quantized_tflite_model)

with open(float_file, 'wb') as f:
  f.write(float_tflite_model)

print("Float model in Mb:", os.path.getsize(float_file) / float(2**20))
print("Quantized model in Mb:", os.path.getsize(quant_file) / float(2**20))

with open('model_quan.tflite', 'wb') as f:
  f.write(quantized_tflite_model)

with open('model_float.tflite', 'wb') as f:
  f.write(float_tflite_model)

model.save_weights('./model/ori_model.hdf5',overwrite=True)
q_aware_model.save_weights('./model/quan_model.hdf5',overwrite=True)
