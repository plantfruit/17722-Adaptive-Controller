import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
import tensorflow as tf
from sklearn.datasets import make_classification
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx, wrap_as_onnx_mixin
from onnxruntime import InferenceSession
from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxDiv
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin

# FILENAMES
dir5_1 = "ML Data/dir5_1.txt"
dir5_1_labels = "ML Data/dir5_1_labels.txt"
dir5_2 = "ML Data/dir5_2.txt"
dir5_2_labels = "ML Data/dir5_2_labels.txt"
dir5_2_48to150 = "ML Data/dir5_2_48to150.txt"
dir5_2_smooth3 = "ML Data/dir5_2_smooth3.txt"
dir5_3_triangle = "ML Data/dir5_3_triangle.txt"
dir6_1 = "ML Data/dir6_1.txt"
dir6_1_labels = "ML Data/dir6_1_labels.txt"
dir6_1_smooth12 = "ML Data/dir6_1_smooth12.txt"
dir6_2 = "ML Data/dir6_2.txt" # Asymmetrical shapes on floor of controller, pre-smoothed by 12
dir6_2_labels = "ML Data/dir6_2_labels.txt"
dir9_123 = "ML Data/dir9_123.txt"
dir9_combined_labels = "ML Data/dir9_combined_labels.txt"
dir9_1 = "ML Data/dir9_1.txt"
dir9_2 = "ML Data/dir9_2.txt"
dir9_3 = "ML Data/dir9_3.txt"
dir9_2and3 = "ML Data/dir9_2and3.txt"
dir9_1_labels = "ML Data/dir9_1_labels.txt"
dir9_2_labels = "ML Data/dir9_2_labels.txt"
dir9_2and3_labels = "ML Data/dir9_2and3_labels.txt"
dir9_combined2_labels = "ML Data/dir9_combined2_labels.txt"
dir9_123_5labels = "ML Data/dir9_123_5labels.txt"
dir9_5label_labels = "ML Data/dir9_5label_labels.txt"

# SELECT FILENAMES FOR ANALYSIS
fileName = dir9_123_5labels
labelFileName = dir9_5label_labels

testFileName = dir9_2and3
testLabelFileName = dir9_2and3_labels

# PARAMETERS
num_labels = 6
files_per_label = 20 
rows_per_file = 1 # Pulses extracted via cross-correlation
kFoldNum = 5 # For cross-validation
labelFontsize = 32 # For the confusion matrix axes
textFontsize = 26 #26
splitNum = 20 # Index to split for train-test split

# PARAMETERS
kFoldOrNot = True # True - Kfold cross validation, otherwise do a normal train-test split
internalSplit = True # True - Split data into train and test sets, False - Load different datasets for train and test
stringLabel = False # False - Numerical labels on the confusion matrix figure
floatLabel = False # Handle edge case where labels are decimals
convertModel = False # Convert trained model to different format for deployment on Android. Don't do this with cross-validation

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
        train_indices.extend(label_rows)
        test_indices.extend(label_rows)
        #if (not(splitNum == files_per_label)):
        #    # Split the indices: first 80 for training, last 20 for testing
        #    train_indices.extend(label_rows[:splitNum])
        #    test_indices.extend(label_rows[splitNum:])
        #else:
        #    train_indices.extend(label_rows[:splitNum])
        #    test_indices.extend(label_rows[:splitNum])        

    # Convert to arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    #print(train_indices)
    #print(test_indices)

    # Split the dataset
    X_train, X_test = X_reshaped[train_indices], X_reshaped[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print(np.shape(X_train))

# Train the SVM model
model = SVC(kernel='linear')
#model = XGBClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier(n_neighbors=5)
#model = DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=100)

# Perform 5-fold cross-validation
if (kFoldOrNot):
    y_pred = cross_val_predict(model, X_reshaped, y, cv=kFoldNum)

    accuracy = accuracy_score(y, y_pred)
    cv_scores = cross_val_score(model, X_reshaped, y, cv=kFoldNum)
    print(cv_scores)
    print(np.mean(cv_scores))
else:
    # Test-train within the same dataset
    if (internalSplit):
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
    # Train with one dataset and test with another dataset
    else:
        X_train = np.loadtxt(fileName)
        y_train = np.loadtxt(labelFileName)
        X_test = np.loadtxt(testFileName)
        y_test = np.loadtxt(testLabelFileName)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Calculate the accuracy of the predictions
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

# Generate the confusion matrix with fixed size
if (not(stringLabel)):
    all_labels = np.arange(1, num_labels + 1)
    #all_labels = (np.arange(1, num_labels + 1) * 0.5 + 0.5 )* 10
else:
    all_labels = ["Stylus", "Screwdriver", "Battery", "Plug", "Motor", "Tripod"]

if (kFoldOrNot):
    cm = confusion_matrix(y, y_pred, labels=all_labels)
else:
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)

def predict_with_onnxruntime(onx, X):
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: X.astype(np.float32)})
    return res[0]

if (convertModel):
    # Code taken from documentation
    # Convert scikit-learn model to ONNX format
    onnx_model = to_onnx(model, X.astype(np.float32), target_opset = 12)
    #ONNX verification
    y_pred = predict_with_onnxruntime(onnx_model, X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # Export to ONNX
    #initial_type = [('input', FloatTensorType([None, 150]))]  # match your vector length
    #onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save to file
    with open("svm_model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    
# Visualize the confusion matrix
fig = plt.figure(figsize=(12, 9))
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_labels, yticklabels=all_labels,
            annot_kws={"size": textFontsize}, vmax = files_per_label * rows_per_file)
# use matplotlib.colorbar.Colorbar object
cbar = ax.collections[0].colorbar
# here set the labelsize by 20
cbar.ax.tick_params(labelsize=textFontsize)
#plt.title('Confusion Matrix (Fixed Size)')
plt.xlabel('Predicted', fontsize = labelFontsize)
plt.ylabel('True', fontsize = labelFontsize)
if (stringLabel):
    textRot = -30
    plt.xticks(fontsize = textFontsize, rotation= textRot, ha='left')
else:
    textRot = 0
    plt.xticks(fontsize = textFontsize)
plt.yticks(fontsize = textFontsize, rotation = 0)#, rotation= 30, ha='right')
plt.tight_layout()
plt.savefig('figure1.pdf', bbox_inches='tight')
plt.show()
