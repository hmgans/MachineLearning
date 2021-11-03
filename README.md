This is a machine learning library developed by Hank Gansert forCS5350/6350 in University of Utah


How to use the Algorithms developed.

Ensemble Learning
All of the files that arent Bagging.py runs the use cases.

All dataframes must be cleaned before putting in any algorithm
cleanNumbericalValues(df)

Input a dataframe, target column, tree limit, gaint type, and number of iterations(bags)
constructAdaTree(df, targetColumn, limit, gainType, T):

Input a dataframe, target column, tree limit, gaint type, and number of iterations(bags)
constructBaggedTree(df, targetColumn, limit, gainType, T)

Input a dataframe, target column, tree limit, gaint type, and number of iterations(bags). Output is a tree
constructRandomForestTree(df, targetColumn, limit, gainType, T)



Linear Regression

Input a dataframe, no need to clean, the r step size, and the limit of iterations.
gradientDecentMethod(df, r, t_limit):
Output is the weight vector

Input a dataframe, no need to clean, the r step size, and the limit of iterations.
stochasticGradientDescent(df, r, t_limit)
Output is the weight vector

Input output vector and test dataframe. Returns cost.
costFunctionBeforeCleaning(vector, df)





