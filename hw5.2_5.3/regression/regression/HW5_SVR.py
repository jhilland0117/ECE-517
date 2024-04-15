import pandas as pd
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.metrics import mean_squared_error

csv = np.genfromtxt ("data.csv", delimiter=",")

Xtest=csv[:,2:21]
Xtrain=csv[:,21:40]
ytest=csv[:,40]
ytrain=csv[:,41]

classifier=SVR(gamma='scale', C=6, epsilon=0.001)
classifier.fit(Xtrain,ytrain)
prediction=classifier.predict(Xtest)
score = classifier.score(Xtrain, ytrain)
MSE = np.square(np.subtract(ytest,prediction))

print(score)

accuracy=cross_val_score(estimator=classifier,X=Xtrain,y=ytrain,cv=10)
np.random.seed(0)
temp=np.arange(ytrain.shape[0])
np.random.shuffle(temp)
Xtrain,ytrain=Xtrain[temp],ytrain[temp]
train_score,valid_score= validation_curve(SVR(),Xtrain,ytrain,param_name="gamma",param_range=np.logspace(-1,3,3),cv=7)

plt.scatter(Xtrain, Xtest, color='red')
plt.plot(prediction)
plt.plot(ytest)
plt.legend(['DATA','Real'])
plt.show()

plt.plot(prediction)
plt.plot(ytest)
plt.legend(['Prediction','Real'])
plt.savefig('SVR_prediction.png')
plt.show()

print(score)
plt.plot(valid_score)
plt.legend(['Valid Score'])
plt.savefig('SVR_validation.png')
plt.show()

plt.plot(MSE)
plt.legend(['Squared Error'])
plt.savefig('SVR_squarederror.png')
plt.show()