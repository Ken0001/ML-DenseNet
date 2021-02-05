"""
    DIVC KEN 2020
    Traning model on our Pomelo dataset
    Support single-label and multi-label
"""
from argparse import ArgumentParser
import read_data as rd
import os
from sklearn.model_selection import train_test_split
from ML_DenseNet import mldensenet

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score

### Get argument
parser = ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset path", dest="dataset", default="None")
parser.add_argument("-m", "--model", help="ML-DenseNet type (0~5)", dest="model", type=int, default="0")
parser.add_argument("-e", "--epoch", help="Epoch", dest="epoch", type=int, default="90")
parser.add_argument("-b", "--batch_size", help="Batch size", dest="batch_size", type=int, default="16")
args = parser.parse_args()

path = args.dataset
ml = False
activation = "softmax"
loss = "categorical_crossentropy"
# Check single-label or multi-label
if "multi" in os.listdir(path+"/train"):
    ml = True
    activation = "sigmoid"
    loss = "binary_crossentropy"

print("|-------------Training info-------------")
print("|-Dataset:   ", args.dataset)
print("|-Model:     ", args.model)
print("|-Epoch:     ", args.epoch)
print("|-Batch_size:", args.batch_size)
print("|-Activation:", activation)
print("|-Loss:      ", loss)
print("|---------------------------------------")

### Read data
print("> Loading training data")
x_train, y_train = rd.read_dataset(path+"/train/*")
print("> Loading testing data")
x_test, y_test = rd.read_dataset(path+"/test/*")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)


print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)
print("Done!")

### Prepare model
img_shape = (224, 224, 3)
num_class = 5

model = mldensenet(img_shape, num_class, mltype=args.model, finalAct=activation)

opt = SGD(lr=0.01, decay=0.0001, momentum=0.9, nesterov=True)

model.compile(loss=loss,
              optimizer=opt,
              metrics=["binary_accuracy", "categorical_accuracy"])



reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, mode='auto', cooldown=3, min_lr=0.00001)

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    if epoch % 30 == 0 and epoch:
        return lr * decay_rate
    if epoch % 60 == 0 and epoch:
        return lr * decay_rate
    return lr

callbacks = [
    reduce_lr
    #LearningRateScheduler(lr_scheduler, verbose=1)
]

### Training model
train_history = model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epoch,
          verbose=1,
          callbacks=callbacks,
          shuffle=True,
          validation_data=(x_val, y_val))


model.save("./model/ml_densenet.h5")

### Evaluate training result
scores = model.evaluate(x_test, y_test, verbose=0)
print('\nTesting result:')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

### Plot training curve
acc = train_history.history['categorical_accuracy']
val_acc = train_history.history['val_categorical_accuracy']
loss = train_history.history['loss']
val_loss = train_history.history['val_loss']
p_epochs = range(1, len(acc) + 1)

plt.plot(p_epochs, acc, 'b', label='Training accurarcy')
plt.plot(p_epochs, val_acc, 'g', label='Validation accurarcy')
plt.plot(p_epochs, loss, 'r', label='Training loss')
plt.plot(p_epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and Validation')
plt.show()