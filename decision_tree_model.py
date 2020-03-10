# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 18:22:17 2020

@author: chint
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('Position_Salaries.csv')
x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
d=DecisionTreeRegressor(criterion='mse',random_state=0)
d.fit(x,y)

y_predict=d.predict(np.array([[6.5]]))


"""graph is wrong ! here the there are no intermediate values in between different 
-values of x,so we are getting a graph like this ! actual graph should contain a discontinuous graph"""
plt.scatter(x,y,color='red')
plt.plot(x,d.predict(x))
plt.title('Salaries v/s positions')
plt.xlabel('Salaries')
plt.ylabel('Positions')
plt.show()


# correct graph
x_grid=np.arange(min(x),max(x),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,d.predict(x_grid))
plt.title('Salaries v/s positions')
plt.xlabel('Salaries')
plt.ylabel('Positions')
plt.show()



