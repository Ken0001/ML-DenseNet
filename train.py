"""
    DIVC KEN 2020
    Traning model on our Pomelo dataset
    Support single-label and multi-label
"""
import click
import os

@click.command()
@click.option('-d', '--dataset', "dataset", help='Dataset', required=True)
@click.option('-m', '--model', "model", help='Model', required=True)
@click.option('-e', '--epoch', "epoch", help='Epoch', required=True, type=int)
@click.option('-b', '--bs', "bs", help='Batch size', required=True, type=int)

def init(dataset, model, epoch, bs):
    print("Train.py")
    print(f'Dataset     => {dataset or "None"}')
    print(f'Model       => {model or "None"}')
    print(f'Epoch       => {epoch or "None"}')
    print(f'Batch size  => {bs or "None"}')

    print("H")
    loc = dataset + "/train"
    category = os.listdir(loc)
    print("Category:", category)
    # a(dataset)
    # b(dataset)
    # c(dataset)
    # d(model, dataset)


def a(dataset):
    print("T", dataset)

def b(dataset):
    print("T", dataset)

def c(model):
    return 0

def d(model, dataset):
    print("T", dataset)
    return 0


if __name__ == '__main__':
    init()
    print("A")
    a()
### Prepare data

# Check single-label or multi-label
# Read data




### Prepare model


### Training model


### Evaluate training result
