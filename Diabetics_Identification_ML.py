#Import Libraries
import numpy as np   #data loading small datasets  matrix  3x3 , 5x5, 10x10 array 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix  #Confusion matrix
from sklearn.metrics import roc_curve     # Receiver Operating Charcteristics 
import seaborn as sns
import matplotlib.pyplot as plt

// reading CSV file
workloads  = pd.read_csv("C:/Users/sonal/Downloads/pima-indians-diabetes.csv")
workloads.head()
workloads.hist()
X = workloads.iloc[:,0:8]
Y = workloads.iloc[:,8].values

#Splitting whole dataset to training and testing dataset
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.20)

#Implementing DecisionTreeClassifier
Dc = DecisionTreeClassifier(max_depth = 150 ,random_state = 1)
Dc.fit(train_x,train_y)
y_pred3 = Dc.predict(test_x)

#Accuracy
print("Accuracy Score of Random Forest Classifier : ", accuracy_score(y_pred3,test_y))

#Confusion Matrix
cMatrix = confusion_matrix(test_y, y_pred3)
print(cMatrix)
ax = sns.heatmap(cMatrix, annot=True, xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['NoDiabetes', 'Diabetes'], cbar=False, cmap='Blues')
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Actual Value")

#ROC Curve
yTestPredictProbability = Dc.predict(test_x)
FPR, TPR, _ = roc_curve(test_y, yTestPredictProbability)
plt.plot(FPR, TPR)
plt.plot([0,1], [0,1],"--", color = "black")
plt.title('ROC Curve')
plt.xlabel('False-Positive Rate')
plt.ylabel('True-Positive Rate')
