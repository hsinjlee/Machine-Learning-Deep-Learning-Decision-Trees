import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_validation
from sklearn.model_selection import cross_val_predict


data = pd.read_csv("C:\\Users\\User\\Desktop\\iris_data.csv")

print(data)
data.features = data[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
data.targets = data.Class 

feature_train, feature_test, target_train, target_test = train_test_split(data.features, data.targets, test_size=.2)

model = DecisionTreeClassifier(criterion='gini')
# model = DecisionTreeClassifier(criterion='entropy')
model.fitted = model.fit(feature_train, target_train)
model.predictions = model.fitted.predict(feature_test)

print(confusion_matrix(target_test, model.predictions))
print(accuracy_score(target_test, model.predictions))

# predicted = cross_validation.cross_val_predict(model,data.features,data.targets, cv=10)
predicted = cross_val_predict(model,data.features,data.targets, cv=10)
print(accuracy_score(data.targets,predicted))


#      SepalLength  SepalWidth  PetalLength  PetalWidth           Class
# 0            5.1         3.5          1.4         0.2     Iris-setosa
# 1            4.9         3.0          1.4         0.2     Iris-setosa
# 2            4.7         3.2          1.3         0.2     Iris-setosa
# 3            4.6         3.1          1.5         0.2     Iris-setosa
# 4            5.0         3.6          1.4         0.2     Iris-setosa
# ..           ...         ...          ...         ...             ...
# 145          6.7         3.0          5.2         2.3  Iris-virginica
# 146          6.3         2.5          5.0         1.9  Iris-virginica
# 147          6.5         3.0          5.2         2.0  Iris-virginica
# 148          6.2         3.4          5.4         2.3  Iris-virginica
# 149          5.9         3.0          5.1         1.8  Iris-virginica
#
# [150 rows x 5 columns]
# [[10  0  0]
#  [ 0  8  1]
#  [ 0  1 10]]
# 0.9333333333333333
# 0.96

#      SepalLength  SepalWidth  PetalLength  PetalWidth           Class
# 0            5.1         3.5          1.4         0.2     Iris-setosa
# 1            4.9         3.0          1.4         0.2     Iris-setosa
# 2            4.7         3.2          1.3         0.2     Iris-setosa
# 3            4.6         3.1          1.5         0.2     Iris-setosa
# 4            5.0         3.6          1.4         0.2     Iris-setosa
# ..           ...         ...          ...         ...             ...
# 145          6.7         3.0          5.2         2.3  Iris-virginica
# 146          6.3         2.5          5.0         1.9  Iris-virginica
# 147          6.5         3.0          5.2         2.0  Iris-virginica
# 148          6.2         3.4          5.4         2.3  Iris-virginica
# 149          5.9         3.0          5.1         1.8  Iris-virginica
#
# [150 rows x 5 columns]
# [[ 8  0  0]
#  [ 0 10  1]
#  [ 0  0 11]]
# 0.9666666666666667
# 0.9533333333333334
