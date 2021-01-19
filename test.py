from tensorflow.keras.models import load_model

model = load_model("./model/ml_densenet.h5")
# Testing on testing data
print("\nTesting on testing data...")
y_pred = model.predict(x_test)
threshold = 0.4
pred = np.zeros(np.array(y_test).shape, dtype=np.int)
np.set_printoptions(precision=6, suppress=True)
for i in range(len(y_pred)):
    for j in range(5):
        if y_pred[i][j] >= threshold:
            pred[i][j] = 1
        else:
            pred[i][j] = 0
    print("Sample ", i)
    print("> Pred: ", y_pred[i])
    print("> Pred: ", pred[i])
    print("> True: ", y_test[i])

true = np.array(y_test)
acc = accuracy_score(true, pred)
scores = model.evaluate(x_test, y_test, verbose=0)

print('\nEvaluate result:')
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print('Test normal accuracy:', acc)