from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pickle

# For this example, we are using a dataset that is included
# in the SKLearn module. This data set has two target classes:
# [malignant] and [benign], and 569 samples with 30 dimensions.

# The load_breast_cancer function returns a dictionary.
data = load_breast_cancer()

# We first must create a Decision Tree object. This is being
# called from the SKLearn module, and was imported on line 3.
model = DecisionTreeClassifier(
    random_state=42, 
    class_weight='balanced')

# From the {data} dictionary, we an create Pandas dataframes. 
# Column names must be added on explicitly, becauase {data} 
# contains Numpy arrays, which have no column names.
X = pd.DataFrame(data['data'], columns=data['feature_names'])
y = pd.Series(data['target'])

# We first set aside a portion of our data as 'Holdout Data'.
# This subset will be used at the end to verify the accuracy
# of our model.
holdout_X, train_X, holdout_y, train_y = (
    train_test_split(X, y, random_state=42,
    shuffle=True, stratify=y)
)

# In this example, we will be evaluating our model with three
# scoring metrics:
accuracy = float()
precision = float()
recall = float()

# In this for loop, we are testing our model after training on
# the training data only (NOT the holdout data). KFold splits
# the data into additional subsets to give us a more accurate
# read on our model's score.
for train_index, test_index in KFold(n_splits=5).split(train_X):
    model.fit(train_X.iloc[train_index], train_y.iloc[train_index])

    y_pred = model.predict(train_X.iloc[test_index])
    y_true = train_y.iloc[test_index]

    accuracy += accuracy_score(y_true, y_pred)
    precision += precision_score(y_true, y_pred)
    recall += recall_score(y_true, y_pred)

# After running KFold, we must devide by the number of folds to get
# an average for each score.
accuracy /= 5
precision /= 5
recall /= 5

# Here we are printing the testing scores of our model after using
# KFold. Before moving on, we will tune the model's hyper-parameters.
print(f'Training scores:\nACC = {accuracy}\nPRE = {precision}\nREC = {recall}')

# Once we have gained as much accuracy as possible from tuning the 
# model, we will score our model one final time using the holdout
# data we set aside at the beginning.
model.fit(train_X, train_y)
pred_y = model.predict(holdout_X)

accuracy = accuracy_score(holdout_y, pred_y)
precision = precision_score(holdout_y, pred_y)
recall = recall_score(holdout_y, pred_y)

# These are the most accurate scores we can get about our model.
print(f'Final Scores:\nACC = {accuracy}\nPRE = {precision}\nREC = {recall}')

# Even though it is necessary to leave the holdout data aside 
# when scoring, we can improve our model further by adding it
# back in for the final fitting.
model.fit(X, y)

# Saving the model is easy using the pickle module!
# load with [model = pickle.loads(filename)]
pickle.dump(model, open('BC_Model.pickle', 'wb'))
print('Model written to "BC_Model.pickle"')


