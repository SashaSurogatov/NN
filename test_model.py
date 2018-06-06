from keras.models import Model, load_model
from keras.optimizers import SGD
import os
import fttlutils
from fttlutils import MODEL_DIR
from fttlutils import mode

#Xtest, Ytest = fttlutils.get_data("/home/sasha/Downloads/test.txt", True)
Xtest, Ytest = fttlutils.get_data("/home/sasha/Downloads/test_skolioz.txt", False)

sgd = SGD(lr=1e-4, momentum=0.9)
# load best model and evaluate
model = load_model(os.path.join(MODEL_DIR, "best-model-VGG16.h5"))
#if mode == 0:
#    model = load_model(os.path.join(MODEL_DIR, "best-model-VGG16.h5"))
#elif mode == 1:
#    model = load_model(os.path.join(MODEL_DIR, "best-model-inception.h5"))
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

Ytest_ = model.predict(Xtest)
#Ytest_ = model.predict_generator(fttlutils.get_data_generator("/home/sasha/Downloads/test.txt"), steps=300)
ytest = Ytest[:len(Ytest_)].argmax(axis=-1)
ytest_ = Ytest_.argmax(axis=-1)
fttlutils.print_stats(ytest, ytest_, "Best Model (FT#1)")