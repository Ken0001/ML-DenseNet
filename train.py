"""
    DIVC KEN 2020
    Traning model on our Pomelo dataset
    Support single-label and multi-label
"""
from argparse import ArgumentParser
import read_data as rd
import os
from sklearn.model_selection import train_test_split

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset", dest="dataset", default="none")
parser.add_argument("-m", "--model", help="Model", dest="model", default="none")
parser.add_argument("-e", "--epoch", help="Epoch", dest="epoch", type=int, default="90")
parser.add_argument("-b", "--batch_size", help="Batch size", dest="batch_size", type=int, default="16")
args = parser.parse_args()
print("|-------------Training info-------------")
print("|-Dataset:   ", args.dataset)
print("|-Model:     ", args.model)
print("|-Epoch:     ", args.epoch)
print("|-Batch_size:", args.batch_size)

path = args.dataset
ml = False
activation = "softmax"
loss = "categorical_crossentropy"
# Check single-label or multi-label
if "multi" in os.listdir(path+"/train"):
    ml = True
    activation = "sigmoid"
    loss = "binary_crossentropy"

print("|-Activation:", activation)
print("|-Loss:      ", loss)
print("|---------------------------------------")
# Read data
print("Loading training data")
x_train, y_train = rd.read_dataset(path+"/train/*")
print("Loading testing data")
x_test, y_test = rd.read_dataset(path+"/test/*")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle= True)


print("Train:", x_train.shape, y_train.shape)
print("Val:", x_val.shape, y_val.shape)
print("Test:", x_test.shape, y_test.shape)
print("Done!")

### Prepare model


### Training model


### Evaluate training result
