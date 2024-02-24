# Importing the necessary libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Loading the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Standardizing the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Creating an SVM classifier
svm = SVC(kernel='linear', C=1.0, random_state=42)

# Training the classifier
svm.fit(X_train_std, y_train)

# Making predictions
y_pred = svm.predict(X_test_std)

# Calculating the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
