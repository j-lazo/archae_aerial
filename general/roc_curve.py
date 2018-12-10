from sklearn.metrics import roc_curve
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import auc



def load_predictions(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(row[2])

            image_name.append(row[0])
    return labels, image_name


def load_labels(csv_file):
    labels = []
    image_name = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            labels.append(float(row[0]))
            image_name.append(row[1])
    return labels, image_name


y_results, names = load_predictions('Inception_predictions.csv')
y_2test, names_test = load_labels('Real_values_test.csv')
y_test = []
y_pred = []

print(len(y_results), len(names))
print(len(y_2test), len(names_test))

for i, name in enumerate(names):
    for j, other_name in enumerate(names_test):
        if name == other_name:
            y_pred.append(float(y_results[i]))
            y_test.append(int(y_2test[j]))

print(len(y_pred))
print(len(y_test))
print(y_pred)
print(y_test)
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)

auc_keras = auc(fpr_keras, tpr_keras)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')

# Zoom in view of the upper left corner.
plt.figure()
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
#plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve (zoomed in at top left)')
plt.legend(loc='best')
plt.show()
