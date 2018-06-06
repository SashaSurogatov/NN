from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import os

import fttlutils

################################# main #################################
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = fttlutils.NUM_CLASSES

np.random.seed(42)

# build model

# (1) instantiate VGG16 and remove top layers
vgg16_model = VGG16(weights="imagenet", include_top=True)
# visualize layers
#print("VGG16 model layers")
#for i, layer in enumerate(vgg16_model.layers):
#    print(i, layer.name, layer.output_shape)
#
#(0, 'input_6', (None, 224, 224, 3))
#(1, 'block1_conv1', (None, 224, 224, 64))
#(2, 'block1_conv2', (None, 224, 224, 64))
#(3, 'block1_pool', (None, 112, 112, 64))
#(4, 'block2_conv1', (None, 112, 112, 128))
#(5, 'block2_conv2', (None, 112, 112, 128))
#(6, 'block2_pool', (None, 56, 56, 128))
#(7, 'block3_conv1', (None, 56, 56, 256))
#(8, 'block3_conv2', (None, 56, 56, 256))
#(9, 'block3_conv3', (None, 56, 56, 256))
#(10, 'block3_pool', (None, 28, 28, 256))
#(11, 'block4_conv1', (None, 28, 28, 512))
#(12, 'block4_conv2', (None, 28, 28, 512))
#(13, 'block4_conv3', (None, 28, 28, 512))
#(14, 'block4_pool', (None, 14, 14, 512))
#(15, 'block5_conv1', (None, 14, 14, 512))
#(16, 'block5_conv2', (None, 14, 14, 512))
#(17, 'block5_conv3', (None, 14, 14, 512))
#(18, 'block5_pool', (None, 7, 7, 512))
#(19, 'flatten', (None, 25088))
#(20, 'fc1', (None, 4096))
#(21, 'fc2', (None, 4096))
#(22, 'predictions', (None, 1000))

# (2) remove the top layer
base_model = Model(input=vgg16_model.input, output=vgg16_model.get_layer("block5_pool").output)

# (3) attach a new top layer
base_out = base_model.output
base_out = Reshape((25088,))(base_out)
top_fc1 = Dense(256, activation="relu")(base_out)
top_fc1 = Dropout(0.5)(top_fc1)
# output layer: (None, 5)
top_preds = Dense(NUM_CLASSES, activation="softmax")(top_fc1)

# (4) freeze weights until the last but one convolution layer (block4_pool)
for layer in base_model.layers[0:14]:
    layer.trainable = False

# (5) create new hybrid model
model = Model(input=base_model.input, output=top_preds)

# (6) compile and train the model
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

best_model = os.path.join(fttlutils.MODEL_DIR, "ft-dl-model-best.h5")
checkpoint = ModelCheckpoint(filepath=best_model, verbose=1,
                             save_best_only=True)

data_gen = fttlutils.get_data_generator("/home/sasha/Downloads/train_korni.txt")
val_gen = fttlutils.get_data_generator("/home/sasha/Downloads/train_korni.txt")
history = model.fit_generator(data_gen, epochs=NUM_EPOCHS,
                    steps_per_epoch=25817/BATCH_SIZE * 0.9,
                    validation_data=val_gen,validation_steps=25817/BATCH_SIZE * 0.1,
                    callbacks=[checkpoint])
fttlutils.plot_loss(history)

# evaluate final model
#Ytest_ = model.predict(Xtest)
#ytest = Ytest.argmax(axis=-1)
#ytest_ = Ytest_.argmax(axis=-1)
#fttlutils.print_stats(ytest, ytest_, "Final Model (FT#1)")
#model.save(os.path.join(fttlutils.MODEL_DIR, "ft-dl-model-final.h5"))
