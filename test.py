from tensorflow.keras.models import load_model
import read_data as rd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-d", "--dataset", help="Dataset path", dest="dataset", default="None")
parser.add_argument("-m", "--model", help="Model path", dest="model", default="None")
args = parser.parse_args()

ml = False
# Check single-label or multi-label
if "multi" in os.listdir(args.dataset+"/train"):
    ml = True

def precision_score(true, pred):
    sample = 0
    score = 0
    for i in range(len(pred)):
        sample += 1
        #print(pred[i])
        #print(true[i])
        #print("--------")
        sample_acc = accuracy_score(true[i], pred[i])
        score += sample_acc
        
    precision = score/sample
    return precision


print("> Loading testing data")
x_test, y_test = rd.read_dataset(args.dataset+"/test/*")

### load model
model = load_model(args.model)
# Testing on testing data
print("\nTest on testing data...")
y_pred = model.predict(x_test)
if ml == True:
    threshold = 0.4
    pred = np.zeros(np.array(y_test).shape, dtype=np.int)
    np.set_printoptions(precision=6, suppress=True)
    for i in range(len(y_pred)):
        for j in range(5):
            if y_pred[i][j] >= threshold:
                pred[i][j] = 1
            else:
                pred[i][j] = 0
        # print("Sample ", i)
        # print("> Pred: ", y_pred[i])
        # print("> Pred: ", pred[i])
        # print("> True: ", y_test[i])
    pre = precision_score(true, pred)
    acc = accuracy_score(true, pred)
    print("Precision score:", pre)
    print("Accuracy score: ", acc)
elif ml == False:
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
    true = np.argmax(y_test, axis=1)
    acc = accuracy_score(true, pred)
    print("Accuracy score:", acc)
