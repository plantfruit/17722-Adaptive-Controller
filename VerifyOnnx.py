import onnxruntime as ort
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# FILENAMES
dir5_1 = "ML Data/dir5_1.txt"
dir5_1_labels = "ML Data/dir5_1_labels.txt"

# SELECT FILENAMES FOR ANALYSIS
fileName = dir5_1
labelFileName = dir5_1_labels

#testFileName = trimic1_3
#testLabelFileName = trimic1relabels

# PARAMETERS
num_labels = 5
files_per_label = 10
rows_per_file = 10 
kFoldOrNot = False # True - Kfold cross validation, otherwise do a normal train-test split
kFoldNum = 5
internalSplit = True
stringLabel = False # False - Numerical labels
floatLabel = False
convertModel = True # Convert trained model to different format for deployment on Android. Don't do this with cross-validation
labelFontsize = 32
textFontsize = 26 #26

train_indices = []
test_indices = []
total_files = num_labels * files_per_label

# Read features and labels from file
X = np.loadtxt(fileName)
#print(np.shape(X))
if (stringLabel):
    y = np.loadtxt(labelFileName, dtype = str)
elif (floatLabel):
    y = np.loadtxt(labelFileName) * 10
else:
    y = np.loadtxt(labelFileName)

# Reshape 1 column/1 row files 
if X.ndim == 1:
    X_reshaped = X.reshape(-1, 1)
else:
    X_reshaped = X

if (not(kFoldOrNot)):
    for label in range(1, num_labels + 1):
        # Get all rows for this label
        label_rows = np.where(y == label)[0]
        #np.where(y == label, 1)[0]

        # Split the indices: first 80 for training, last 20 for testing
        train_indices.extend(label_rows[:80])
        test_indices.extend(label_rows[80:])

    # Convert to arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    #print(train_indices)
    #print(test_indices)

    # Split the dataset
    X_train, X_test = X_reshaped[train_indices], X_reshaped[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

#===========================================================================================================================
# ONNX Verification
#===========================================================================================================================

# Load the ONNX model
sess = ort.InferenceSession("svm_model.onnx")

# Get the model's input name (usually 'input')
input_name = sess.get_inputs()[0].name

# Run predictions
preds = []
for x in X_test:
    input_data = x.astype(np.float32).reshape(1, -1)  # Must match (1, 150)
    result = sess.run(None, {input_name: input_data})
    
    output = result[0]  # shape (1, 1) or (1, n_classes) depending on your model
    label = output#int((output[0][0] >= 0.5))  # If it's probability output
    preds.append(label)

# Evaluate
acc = accuracy_score(y_test, preds)
print(f"ONNX Model Accuracy: {acc * 100:.2f}%")
