import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# Reading the file
train_data = pd.read_csv('./train_data.csv', index_col="job_id")
test_data = pd.read_csv("./test_data_unlabeled.csv", index_col="job_id")

# Defining X and Y
X = train_data[['memory_GB', 'network_log10_MBps',
          'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]
Y = train_data['failed']
test_X = test_data[['memory_GB', 'network_log10_MBps',
                   'local_IO_log10_MBps', 'NFS_IO_log10_MBps']]

#Splitting Data
train_x, test_x, train_y,test_y = train_test_split(X, Y, train_size=0.5, test_size=0.5, random_state=0)


#PCA
pca = PCA(n_components=3)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)
print(pca.explained_variance_ratio_)

# Model
myModel = DecisionTreeClassifier(random_state=0)
myModel.fit(X, Y)
predicted_y = myModel.predict(test_X)

# Confusion Matrix
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(test_y, predicted_y)
print(cm)

print(
    mean_absolute_error(test_y, predicted_y))

score = accuracy_score(test_y, predicted_y)
print(score)


# Writing to csv
output = pd.DataFrame({'job_id': test_X.index,
                        'failed': predicted_y})
output.to_csv('model_abridged_test.csv', index=False)
