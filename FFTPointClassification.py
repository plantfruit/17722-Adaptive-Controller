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

# FILENAMES
# Old data, 3 x 3 grid. Each microphone was recorded separately
mic1_1 = 'Data/trimic1_1.txt'
mic1_2 = 'Data/trimic1_2.txt'
mic1_3 = 'Data/trimic1_3.txt'
grid9_5samples = 'Data/bal2labels.txt'

# 5 x 5 grid
# Each row is a pulse FFT
# Rows are grouped sequentially by the file they were extracted from
# e.g. 20 rows were from the same file 
trimic1 = 'Data/5by5_trimic1.txt' # 20 pulses per file
trimic1duplicate = 'Data/5by5_trimic1_possibleduplicate.txt'
trimic1labels = 'Data/5by5_trimic1_labels.txt'
trimic1re = 'Data/5x5_trimic1_re.txt' # Only 10 pulses per file
trimic1relabels = 'Data/5by5_trimic1_re_labels.txt'
trimic1_1 = 'Data/5x5_trimic1_1.txt' # Individual microphones' rows
trimic1_2 = 'Data/5x5_trimic1_2.txt'
trimic1_3 = 'Data/5x5_trimic1_3.txt'
trimic1_1and2 = 'Data/5x5_trimic1_1and2.txt' # Remove 1 microphone from the row
trimic1_2and3 = 'Data/5x5_trimic1_2and3.txt'
trimic1_1and3 = 'Data/5x5_trimic1_1and3.txt'
trimic1_1pulse = 'Data/5x5_trimic1_onepulse.txt' # Extract 1 pulse instead of 10 pulses
trimic1_1pulse_labels = 'Data/5x5_trimic1_onepulse_labels.txt'

miscobj1 = 'Data/miscobj3.txt'
miscobj1labels = 'Data/miscobj3_labels.txt'

# Small array with 3 labels, and 3 "pulses per file," that is used to test the grouping function
groupingTest = 'Data/groupsorttest_features.txt'
groupingTestLabels = 'Data/groupsorttest_labels.txt'

# 3x3 grid, pulse FFTs
g3x3_trimic1 = 'Data/3x3_trimic1.txt' # 15 files per label, groups of 5 trials that are "soft, "medium," and "hard" press
g3x3_trimic1_labels = 'Data/3x3_trimic1_labels.txt'

# Regression trials (straight lines)
# Full FFT spectrum
regY1_1 = 'Data/5x5_regY_1.txt'
regY1_2 = 'Data/5x5_regY_2.txt'
regY1_3 = 'Data/5x5_regY_3.txt'
# New balloon, hard force (full FFT spectrum)
regX2_1 = 'Data/5x5_2_regX_1.txt'
regX2_2 = 'Data/5x5_2_regX_2.txt'
regX2_3 = 'Data/5x5_2_regX_3.txt'
regY2_1 = 'Data/5x5_2_regY_1.txt'
regY2_2 = 'Data/5x5_2_regY_2.txt'
regY2_3 = 'Data/5x5_2_regY_3.txt'
regX1_10points_0_3mm = 'Data/regression_10point_0_3res.txt' # Label file
reg_10points_integer = 'Data/regression_10point_integer.txt'
regX1_1_cols57to75 = 'Data/5x5_2_regX_1_cols57to75.txt'
regX1_1_col67 = 'Data/5x5_2_regX_1_col67.txt'

# 1D tube, gripper arm, 6 objects
D1_6obj2 = 'Final Data/1D_6obj2.txt'
D1_6obj2_labels = 'Final Data/1Dtube_6obj2_labels.txt'
D1_6obj2_lessSmooth = 'Final Data/1D_6obj2_smooth3.txt'
D1_6obj2_v3 = 'Final Data/1D_6obj2_smooth1_wind.txt' # Reduced smoothing factor to 1, windowed 2.5 to 15 kHz
# 1 to 9 cm, at 0.5 cm resolution
tube1D_res05 = 'Data/1Dtube_05res.txt'
tube1D_res05_labels = 'Data/1Dtube_05res_labels.txt'

BGwhite_vol1 = 'Data/BGwhite_vol1.txt'
BGwhite_vol2 = 'Data/BGwhite_vol2.txt'
BGwhite_vol3 = 'Data/BGwhite_vol3.txt'
grid3x3_labels  = 'Data/3x3_labels.txt'

# 3x3 grid, 18-20 kHz, linear chirp and exponential chirp
chirpLin = 'Final Data/ChirpLin.txt'
chirpExp = 'Final Data/ChirpExp.txt'
standard3x3_labels = 'Final Data/3x3_labels.txt'

# SELECT FILENAMES FOR ANALYSIS
fileName = chirpExp

labelFileName = standard3x3_labels

testFileName = trimic1_3
 
testLabelFileName = trimic1relabels

# PARAMETERS
num_labels = 9
files_per_label = 10
rows_per_file = 10 
total_files = num_labels * files_per_label
total_rows = total_files * rows_per_file # Unused
kFoldOrNot = True # True - Kfold cross validation, otherwise do a normal train-test split
kFoldNum = 5
internalSplit = True
stringLabel = False # False - Numerical labels
floatLabel = False 
labelFontsize = 32
textFontsize = 26 #26

# Train-test split: First 80 rows/train, last 20 rows/test per label
train_indices = []
test_indices = []

# Read features and labels
X = np.loadtxt(fileName)
#print(np.shape(X))
if (stringLabel):
    y = np.loadtxt(labelFileName, dtype = str)
elif (floatLabel):
    y = np.loadtxt(labelFileName) * 10
else:
    y = np.loadtxt(labelFileName)

if X.ndim == 1:
    X_reshaped = X.reshape(-1, 1)
else:
    X_reshaped = X

groups_per_label = 3
files_per_group = 5

if (not(kFoldOrNot)):
    for label in range(1, num_labels + 1):
        # Get all rows for this label
        label_rows = np.where(y == label)[0]
        #np.where(y == label, 1)[0]

        # These next 2 blocks do the same thing (3x3 grid, varying force, classification)
        
        # Iterate over each group of 5 files
    ##    for group in range(groups_per_label):
    ##        # Start index of this group
    ##        group_start = group * files_per_group * rows_per_file
    ##
    ##        # Indices for this group
    ##        group_indices = label_rows[group_start:group_start + files_per_group * rows_per_file]
    ##
    ##        # First 4 files (40 rows) for training
    ##        train_indices.extend(group_indices[:4 * rows_per_file])
    ##
    ##        # Last file (10 rows) for testing
    ##        test_indices.extend(group_indices[4 * rows_per_file:])

    ##    train_indices.extend(label_rows[:40])
    ##    train_indices.extend(label_rows[50:90])
    ##    train_indices.extend(label_rows[100:140])
    ##    test_indices.extend(label_rows[40:50])
    ##    test_indices.extend(label_rows[90:100])
    ##    test_indices.extend(label_rows[140:150])
        
        # Split the indices: first 80 for training, last 20 for testing
        #train_indices.extend(label_rows[:80])
        #test_indices.extend(label_rows[80:])

        # Reversed order
        train_indices.extend(label_rows[50:])
        test_indices.extend(label_rows[:50])
        
        # Split the indices: 
        # First 20 rows and last 60 rows for training
        #train_indices.extend(label_rows[:50])
        #train_indices.extend(label_rows[100:])
        # 2nd set of 20 rows for testing
        #test_indices.extend(label_rows[50:100])

        # Split the indices: 
        # First 20 rows and last 60 rows for testing
        #test_indices.extend(label_rows[:50])
        #test_indices.extend(label_rows[100:])
        # 2nd set of 20 rows for training
        #train_indices.extend(label_rows[50:100])


    # Convert to arrays for indexing
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    print(train_indices)
    print(test_indices)

    # Split the dataset
    X_train, X_test = X_reshaped[train_indices], X_reshaped[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

# Train the SVM model
#model = XGBClassifier()
#model = GaussianNB()
#model = KNeighborsClassifier(n_neighbors=5)
#model = DecisionTreeClassifier()
#model = RandomForestClassifier(n_estimators=100)
model = SVC(kernel='linear')  # You can change kernel here (e.g., 'rbf', 'poly')

if (kFoldOrNot):
    # Perform cross-validation and get predictions for each sample
    y_pred = cross_val_predict(model, X_reshaped, y, cv=kFoldNum)

    # Print predictions and true labels for each sample
    #for i in range(len(y_pred)):
    #    print(f"Sample {i+1} - Predicted: {y_pred[i]}, True: {y[i]}")

    # Perform 5-fold cross-validation
    accuracy = accuracy_score(y, y_pred)
    cv_scores = cross_val_score(model, X_reshaped, y, cv=kFoldNum)
    print(cv_scores)
    print(np.mean(cv_scores))
else:
    if (internalSplit):
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)
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
