import csv
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np

#input file
data = pd.read_csv('dataset.csv')

# data without class label is taken
data_ = data[data.columns[:30]]

#value of class label is taken
target = data.iloc[:,-1]

#total no of positive and negative value is taken
y_positive_no, y_negative_no = target.value_counts()[1], target.value_counts()[-1]


fold_no = int(len(data_)/5)
k_fold =  5
single_ft = {}
feature_selected = []
max_acc = 0

# the main function that select the best feature based on the SFS wrapper method using stratified 5 fold cross validation and SVM classifier
def feature_selection(feature_selected, max_acc):
	for i in range(len(data_.columns)):
		if i not in feature_selected: 
			list_ = [i]
			columns = list_ + feature_selected
			
			feature = data_.iloc[:,columns]
			acc = 0

			# the main logic for implementing stratified 5 fold cross validation and on each fold training and test data set are taken accordingly.
			for fold in range(k_fold):
				duplicate_data = feature.copy()
				duplicate_target = target.copy()
				test_rows  = []
				test_set = feature[fold*fold_no: fold*fold_no + fold_no]
				test_target = target[fold*fold_no: fold*fold_no + fold_no]
				for t in range(fold*fold_no, fold*fold_no + fold_no):
					test_rows.append(t)
				train_data = duplicate_data.drop(test_rows, axis=0)
				if len(feature_selected) == 0:
					train_data = train_data.to_numpy()

					train_data = np.reshape(train_data, (-1,1))

					test_set = test_set.to_numpy()

					test_set = np.reshape(test_set, (-1,1))

				train_target = duplicate_target.drop(test_rows, axis=0)

				# SVM classifier that taken training data to train and predict value on test data.
				clf = SVC(gamma='auto')
				clf.fit(train_data, train_target)

				y_pred = clf.predict(test_set)
				acc += metrics.accuracy_score(test_target, y_pred)

			single_ft[i] = (acc / k_fold)

	# feature/s with maximum is selected
	maximum = max(single_ft, key=single_ft.get)
	
	# if accuracy is improved compared to previous one, then it is best feature so we will add it in our feature selection list
	if single_ft[maximum] > max_acc:
		max_acc = single_ft[maximum]

		feature_selected.append(maximum)
		print(max_acc, feature_selected)
		feature_selection(feature_selected, max_acc)
	else:
		return feature_selected

feature_selection(feature_selected, max_acc)

# 0.9438263229308006 [7, 13, 14, 5, 1, 27, 8, 23, 28, 2, 6, 25, 24, 11, 16, 0, 15, 18]
# The dataset is large that has 30 features and around 10K instances so it might take time to get the result. here I have shown final list of best features that gives me 94.38% accuracy
