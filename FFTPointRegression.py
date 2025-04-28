import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn, to_onnx, wrap_as_onnx_mixin
from onnxruntime import InferenceSession
from skl2onnx.algebra.onnx_ops import OnnxSub, OnnxDiv
from skl2onnx.algebra.onnx_operator_mixin import OnnxOperatorMixin
from create_cnn import *
from sklearn.svm import SVR
import pandas as pd

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
dir6_2 = "ML Data/dir6_2.txt"
dir6_2_labels = "ML Data/dir6_2_labels.txt"

# SELECT FILENAMES FOR ANALYSIS
fileName = dir6_2
labelFileName = dir6_2_labels 

#testFileName = trimic1_3
#testLabelFileName = trimic1relabels

# PARAMETERS
num_labels = 6
files_per_label = 10
rows_per_file = 1
kFoldOrNot = True # True - Kfold cross validation, otherwise do a normal train-test split
kFoldNum = 5
internalSplit = True
stringLabel = False # False - Numerical labels on the confusion matrix figure
floatLabel = False
convertModel = False # Convert trained model to different format for deployment on Android. Don't do this with cross-validation
labelFontsize = 32
textFontsize = 26 #26
splitNum = 10 # Index to split for train-test split

train_indices = []
test_indices = []
total_files = num_labels * files_per_label

def get_feats_from_file(fileName):
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

    df = pd.DataFrame(X)
    df['label'] = y
    df['press_no_press_label'] = np.where(df['label'] == 6, 0, 1)  # Assuming label 6 is "no press"

    #getting train_test_split for the press no press model
    press_no_press_df = df.drop(['press_no_press_label'], axis=1)
    train_press_no_press, test_press_no_press = train_test_split(press_no_press_df)

    #all training data for press_no_press classifier model. Train the final model on this data
    all_press_no_press_x = press_no_press_df.drop(['press_no_press_label'], axis=1).to_numpy()
    all_press_no_press_y = press_no_press_df['press_no_press_label'].to_numpy()

    #training and testing split for press_no_press classifier model. Use for checking accuracy of the model. 
    train_press_no_press_x = train_press_no_press.drop(['press_no_press_label'], axis=1).to_numpy()
    train_press_no_press_y = train_press_no_press['press_no_press_label'].to_numpy()
    test_press_no_press_x = test_press_no_press.drop(['press_no_press_label'], axis=1).to_numpy()
    test_press_no_press_y = test_press_no_press['press_no_press_label'].to_numpy()

    #getting train_test_split for the y-axis regression model
    #note: change this if we are no longer using 6 as the label for the "no press" state
    use_df = df[df['label'] != 6]
    train, test = train_test_split(use_df)

    #all training data for y-axis regression model. Train the final model on this data
    all_x = use_df.drop(['label'], axis=1).to_numpy()
    all_y = use_df['label'].to_numpy()

    #training and testing split for y-axis regression model. Use for checking accuracy of the model. 
    train_x = train.drop(['label'], axis=1).to_numpy()
    train_y = train['label'].to_numpy()
    test_x = test.drop(['label'], axis=1).to_numpy()
    test_y = test['label'].to_numpy()
    return all_x, all_y, train_x, train_y, test_x, test_y, all_press_no_press_x , all_press_no_press_y, train_press_no_press_x, train_press_no_press_y, test_press_no_press_x, test_press_no_press_y

all_x, all_y, train_x, train_y, test_x, test_y, all_press_no_press_x , all_press_no_press_y, train_press_no_press_x, train_press_no_press_y, test_press_no_press_x, test_press_no_press_y = get_feats_from_file(fileName)
'''if (not(kFoldOrNot)):
    for label in range(1, num_labels + 1):
        # Get all rows for this label
        label_rows = np.where(y == label)[0]
        #np.where(y == label, 1)[0]

        if (not(splitNum == files_per_label)):
            # Split the indices: first 80 for training, last 20 for testing
            train_indices.extend(label_rows[:splitNum])
            test_indices.extend(label_rows[splitNum:])
        else:
            train_indices.extend(label_rows[:splitNum])
            test_indices.extend(label_rows[:splitNum])        

    # Convert to arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    #print(train_indices)
    #print(test_indices)

    # Split the dataset
    X_train, X_test = X_reshaped[train_indices], X_reshaped[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print(np.shape(X_train))'''

# Train the SVM model
#model = SVC(kernel='linear')
#model = SVC(kernel='rbf')
#model = XGBClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier(n_neighbors=5)
#model = DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=100)
#cnn = SimpleCNN() #can onlu b
#model =  MLPRegressor(random_state=1, max_iter=2000, tol=0.1)
model_x = SVR()
model_y = SVR()

#same model as kevin's initial model
model_press_no_press = SVC(kernel='linear')

#from left to center 
# "1" = (-2, 0)
# "2" = (0, 2)
# "3" = (2, 0)
# "4" = (0, -2)
# "5" = (0,0)
# "1" = pressed

'''
there will need to be a new relabeling of the data, since this is directly mapped to kevin's initial data collection

def relabel_data(y):
    new_x_axis_y = []
    new_y_axis_y = []
    y_no_press = []
    for i in y:
        if i == 1:
            new_x_axis_y.append(-2)
            new_y_axis_y.append(0)
            y_no_press.append(1)
        elif i == 2:
            new_x_axis_y.append(0)
            new_y_axis_y.append(2)
            y_no_press.append(1)
        elif i == 3:
            new_x_axis_y.append(2)
            new_y_axis_y.append(0)
            y_no_press.append(1)
        elif i == 4:
            new_x_axis_y.append(0)
            new_y_axis_y.append(-2)
            y_no_press.append(1)
        elif i == 5:
            new_x_axis_y.append(0)
            new_y_axis_y.append(0)
            y_no_press.append(1)
        else:
            y_no_press.append(0)
    return np.array(new_x_axis_y).reshape(-1, 1), np.array(new_y_axis_y).reshape(-1, 1), np.array(y_no_press).reshape(-1,1)
'''

#TODO: fix this relabeling function such that it works with the labels given in the datafiles. 
def relabel_data_y(x, y):
    no_press_x = []
    no_press_y = []
    y = []
    x =[] 
    return x, y, no_press_x, no_press_y
#train_x_axis_y, train_y_axis_y, train_y_no_press= relabel_data(train_y)
#test_x_axis_y, test_y_axis_y, test_y_no_press= relabel_data(test_y)

train_x, train_y, train_no_press_x, train_no_press_y = relabel_data_x(train_x, train_y)
test_x, test_y, test_no_press_x, test_no_press_y = relabel_data_x(test_x, test_y)


model_y.fit(train_x, train_y)
#model_y.fit(train_x, train_y_axis_y)
model_press_no_press.fit(train_press_no_press_x, train_no_press_y)

# Make predictions on the test set
#y_pred_x = model_x.predict(test_x)
y_pred_y = model_y.predict(test_x)
y_pred_no_press = model_press_no_press.predict(test_press_no_press_x)

accuracy = accuracy_score(test_press_no_press_y, y_pred_no_press)
mse_y = mean_squared_error(test_y, y_pred_y)

print(f'Regression MSE: {mse_y}')
print(f'Classification Accuracy for press no press: {accuracy}')

def display_mse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line
    plt.grid()
    plt.show()

def final_train_and_save_model(model, X, y, model_name):
    if (convertModel):
        # Code taken from documentation
        # Convert scikit-learn model to ONNX format
        onnx_model = to_onnx(model, X.astype(np.float32), target_opset = 12)
        #ONNX verification
        #y_pred = predict_with_onnxruntime(onnx_model, X_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #print(accuracy)

        # Export to ONNX
        #initial_type = [('input', FloatTensorType([None, 150]))]  # match your vector length
        #onnx_model = convert_sklearn(model, initial_types=initial_type)

def predict_with_onnxruntime(onx, X):
    sess = InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    res = sess.run(None, {input_name: X.astype(np.float32)})
    return res[0]





'''
should appear in the phone as
prediction  = model_no_press.predict(feats)
if prediction == 'pressed:
    classification = model_y.predict(feats)
'''
