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
