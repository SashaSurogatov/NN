from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import np_utils
import numpy as np
import os

import fttlutils
from fttlutils import mode
#vgg16 - 0
#inception_v3 - 1
#Xception - 2

if mode == 0:
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg16 import preprocess_input
elif mode == 1:
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input

################################# main #################################
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_CLASSES = fttlutils.NUM_CLASSES

np.random.seed(42)

Xtrain, Ytrain = fttlutils.get_data("/home/sasha/Downloads/train_skolioz.txt")
Xtest, Ytest = fttlutils.get_data("/home/sasha/Downloads/test_skolioz.txt")

print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
# build model

# (1) instantiate VGG16 and remove top layers
model = None
if mode == 0:
    model = VGG16(weights="imagenet", include_top=True)
elif mode == 1:
    model = InceptionV3(weights="imagenet", include_top=True)
# visualize layers
print("model layers")
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape)

# (2) remove the top layer
out_layer = None
if mode == 0:
    out_layer = model.get_layer("block5_pool").output
elif mode == 1:
    out_layer = model.get_layer("mixed10").output

base_model = Model(input=model.input, output=out_layer)

# (3) attach a new top layer
base_out = base_model.output
top_preds = None
if mode == 0:
    base_out = Reshape((25088,))(base_out)
    top_fc1 = Dense(256, activation="relu")(base_out)
    top_fc1 = Dropout(0.5)(top_fc1)
    top_preds = Dense(NUM_CLASSES, activation="softmax")(top_fc1)
elif mode == 1:
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    top_preds = Dense(NUM_CLASSES, activation='softmax')(x)

# (4) freeze weights until the last but one convolution layer (block4_pool)
for layer in base_model.layers:
    layer.trainable = False

# (5) create new hybrid model
model = Model(input=base_model.input, output=top_preds)

# (6) compile and train the model
sgd = SGD(lr=1e-4, momentum=0.9)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

best_model = None
if mode == 0:
    best_model = os.path.join(fttlutils.MODEL_DIR, "best-model-VGG16.h5")
elif mode == 1:
    best_model = os.path.join(fttlutils.MODEL_DIR, "best-model-inception.h5")

checkpoint = ModelCheckpoint(filepath=best_model, verbose=1,
                             save_best_only=True)
history = model.fit([Xtrain], [Ytrain], epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE, validation_split=0.1,
                    callbacks=[checkpoint])
fttlutils.plot_loss(history)

# evaluate final model
Ytest_ = model.predict(Xtest)
ytest = Ytest.argmax(axis=-1)
ytest_ = Ytest_.argmax(axis=-1)
fttlutils.print_stats(ytest, ytest_, "Final Model (FT#1)")
model.save(os.path.join(fttlutils.MODEL_DIR, "ft-dl-model-final.h5"))
