import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model 


"""def linear_regression (x,y):
	diabetes = datasets.load_diabetes()

	# Use only one feature
	#diabetes_X = diabetes.data[:, np.newaxis, 2]

	# Split the data into training/testing sets
	#diabetes_X_train = diabetes_X[:-20]
	#diabetes_X_test = diabetes_X[-20:] 
	diabetes_X_train = x[:-20]
	diabetes_X_test = x[-20:]

	# Split the targets into training/testing sets
	diabetes_y_train = diabetes.target[:-20]
	diabetes_y_test = diabetes.target[-20:]

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(diabetes_X_train, diabetes_y_train)

	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print("Mean squared error: %.2f"
	      % np.mean((regr.predict(diabetes_X_test) - diabetes_y_test) ** 2))
	# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test)) """ 

from numpy import loadtxt, zeros, ones, array, linspace, logspace
from pylab import scatter, show, title, xlabel, ylabel, plot, contour
from random import randint
 
 
print("""
				&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&6&&&&
				$ you will need to insert your data file's directory just like 
				$ " 'C:\\Users\\RACHID\\Desktop\\python\\data.txt' " , and also 
				$ your variable just like " 'population' " , and as for the 
				$ title , it should also be between two commas "'" , so that you
				$ wont have any problems executing the code. and for the time ,
				$ i set the unit in years , if you are counting with months or
				$ days , you can change it.
				&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
""")
 
#generating the data
m = int(input("enter number of training samples you want to work on , prefered to be under 50 : "))
X = []
y = []
for i in range(m) :
	X.append(i)
	y.append(randint(0,90))
 
#Evaluate the linear regression
def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
 
    predictions = X.dot(theta).flatten()
 
    sqErrors = 0
    
    for i in range(m):
        sqErrors += (predictions[i] - y[i]) ** 2
 
    J = (1.0 / (2 * m)) * sqErrors
 
    return J
 
 
def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    J_history = zeros(shape=(num_iters, 1))
 
    for i in range(num_iters):
 
        predictions = X.dot(theta).flatten()
 
        errors_x1 = (predictions - y) * X[:, 0]
        errors_x2 = (predictions - y) * X[:, 1]
 
        theta[0][0] = theta[0][0] - alpha * (1.0 / m) * errors_x1.sum()
        theta[1][0] = theta[1][0] - alpha * (1.0 / m) * errors_x2.sum()
 
        J_history[i, 0] = compute_cost(X, y, theta)
 
    return theta, J_history
 
 
#Plot the data
scatter(X, y, marker='o', c='b')
title("predicttion script")
xlabel("time in minutes")
#show()
 
 
#Add a column of ones to X (interception data)
it = ones(shape=(m, 2))
it[:, 1] = X
 
#Initialize theta parameters
theta = zeros(shape=(2, 1))
 
#Some gradient descent settings
iterations = 2500
alpha = 0.01
 
#compute and display initial cost
J = compute_cost(it, y, theta)
 
theta, J_history = gradient_descent(it, y, theta, alpha, iterations)
 
 
#Predict values for the future
h=input("enter the minute that you want to predict : ")
predict = array([1, 3.5]).dot(theta).flatten()
print ('For the %s minute , we predict a value of %f' %(h,predict))
 
#Plot the results
result = it.dot(theta).flatten()
plot(X , result)
show()
input()

