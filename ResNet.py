from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import tensorflow as tf
# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.05)

# import libraries
import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,GlobalAveragePooling2D, Dropout
from keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_yaml
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print('Download Model...')
base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=3)
x = base_model.output

Epoch = 50
Batch = 16
Aug = 'False'

x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

train_datagen = ImageDataGenerator(
  preprocessing_function=preprocess_input,)
	# zoom_range=0.15,
	# width_shift_range=0.2,
	# height_shift_range=0.2,
	# shear_range=0.15,
	# brightness_range=(1, 1.3),
	# horizontal_flip=True,
	# vertical_flip=True,
	# fill_mode='nearest')


print('Create dataset ...')
train_generator = train_datagen.flow_from_directory('Images/Oversample/Train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=Batch,
                                                 class_mode='categorical',
                                                 shuffle=True)

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
validation_generator = test_datagen.flow_from_directory("Images/Oversample/Test",
                                                        target_size=(224,224),
                                                        color_mode='rgb',
                                                        batch_size=Batch,
                                                        class_mode='categorical',
                                                        shuffle=False)

print('Training ...')              
H = model.fit_generator(generator=train_generator,
                   validation_data=validation_generator,
                   steps_per_epoch=step_size_train,
                   epochs=Epoch)

source = "ResNetModel/"
finishTime = time.strftime("%Y.%m.%d_%H.%M.%S")
os.makedirs(source+finishTime)

N = Epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(source+finishTime+"/loss.png")

plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title('Training Accuracy on Dataset')
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig(source+finishTime+"/acurracy.png")

print('Testing ...')
# file validation picture
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk("Images/Oversample/Test"):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

Y_pred = model.predict_generator(validation_generator, len(files) // Batch +1)

y_pred = np.argmax(Y_pred, axis=1)
y_test = validation_generator.classes

print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = train_generator.class_indices.keys()
print(classification_report(y_test, y_pred, target_names=target_names))
modelAcc = accuracy_score(y_test, y_pred)

# save all model model
# model.save(source+finishTime+'/Inception3.'+str("{:.2f}".format(modelAcc))+'.'+str(Epoch)+'.'+str(Batch)+'.'+Aug+'.h5')

# save model weight
model.save_weights(source+finishTime+'/Inception3.Weight.'+str("{:.2f}".format(modelAcc))+'.'+str(Epoch)+'.'+str(Batch)+'.'+Aug+'.h5')

# save model layers
model_yaml = model.to_yaml()
with open(source+finishTime+'/Inception3.Architecture.'+str("{:.2f}".format(modelAcc))+'.'+str(Epoch)+'.'+str(Batch)+'.'+Aug+'.yaml', 'w') as yaml_file:
    yaml_file.write(model_yaml)
