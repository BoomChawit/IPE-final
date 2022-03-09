import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier   
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import chawit_function as chawit

# 1 Importing Data
training_values = pd.read_csv("sample_data/assignment_2019/training_values.csv")
training_labels = pd.read_csv("sample_data/assignment_2019/training_labels.csv")
cities = pd.read_csv("sample_data/assignment_2019/cities.csv")

# 2 Cleaning Data
training_values.fillna(0)
training_labels.fillna(0)
cities.fillna(0)

# 3 Plot

# 3a
users_locations = training_values[['longitude','latitude','users','gps_height']]
users_locations.columns =  ['longitude','latitude','users','gps_height']
users_locations = users_locations.replace({-2.000000e-08: np.nan,0: np.nan})
users_locations = users_locations.dropna()

plt.figure()
plt.scatter(x=users_locations['longitude'], y= users_locations['latitude'], s=users_locations['users']/50,c='b',marker = 'o')
plt.scatter(x = cities['longitude'],y=cities['latitude'], marker = 'x', color = 'r')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('figure1.png')
plt.show()

#3b
plt.figure()
plt.scatter(x=users_locations['longitude'], y=users_locations['latitude'], 
             s=users_locations['users']/50, c=users_locations['gps_height'], cmap = 'terrain')
plt.scatter(x = cities['longitude'],y=cities['latitude'], marker = 'x', color = 'r')

label = cities['city'].to_numpy()
labelx = cities['longitude'].to_numpy()
labely = cities['latitude'].to_numpy()

for i,type in enumerate(label):
    x = labelx[i]
    y = labely[i]
    plt.text(x+0.3,y+0.3,type, fontsize = 9)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.savefig('figure2.png')
plt.show()

# 4a Extract

# 4b Plot and fit
def fit_graph(x,a,b,c):
    y = a*x**2 +b*x+c
    return y

cities_mean = chawit.users(users_locations,cities)
x = cities['population'].to_numpy()
y = cities_mean

for i in range(0,len(x)):
    for j in range(i,len(x)):
        if x[i] > x[j]:
            constant1 = x[j]
            x[j] = x[i]
            x[i] = constant1
            constant2 = y[j]
            y[j] = y[i]
            y[i] = constant2
print(x)
print(y)

# plt.figure()
# plt.scatter(x,y,label='Data set')
# fit1 = np.polyfit(x,y,1)
# fit_1 = np.poly1d(fit1)(x)
# plt.plot(x,fit_1, label = f'{round(fit1[0],2)}*x + {round(fit1[1],2)}')
# plt.legend(bbox_to_anchor=(1,1), loc="upper left")
# plt.show()

plt.figure()
con1,con2 = curve_fit(fit_graph, x, y)
yfit = fit_graph(x, *con1)
plt.plot(x, y, 'o', label='raw data')
plt.plot(x, yfit, 'r', label= f'{round(con1[0],6)}*x**2 + {round(con1[1],6)}*x + {round(con1[2],2)}')
plt.legend(bbox_to_anchor=(1,1), loc="upper left")
plt.xlabel("Population")
plt.ylabel("Mean number of users of wells within 200km of city centre")
plt.savefig('figure3.png')
plt.show()

# 5 
training_values['decade'] = 10*(training_values['construction_year']//10).astype(int)
training_values['function'] = training_labels['status_group']

def group_aggregate(a,b):
    group = training_values[[a,b]].astype(str)
    group = group.replace({0:np.nan})
    group = group.dropna()
    group['number'] = 1
    group = group.groupby([a,b])['number'].count().unstack().fillna(0).astype(int)
    return group


# 5 case 1
display(group_aggregate('decade','extraction_type'))

# 5 case 2
display(group_aggregate('decade','function'))


# 6 ML
training_values = pd.read_csv("sample_data/assignment_2019/training_values.csv")
training_labels = pd.read_csv("sample_data/assignment_2019/training_labels.csv")

machine = training_values.loc[:,['construction_year','extraction_type','scheme_management','region','basin',
                                 'management_group','water_quality','quality_group','district_code','region_code',
                                 'quantity', 'quantity_group','installer', 
                                 'source', 'source_type','waterpoint_type']]

machine['function'] = training_labels.loc[:,'status_group']
machine = machine.replace({0:np.nan})
machine = machine.dropna()

for feature in machine.columns:
    if feature != 'construction_year':
        a = list(machine.loc[:,feature].unique())
        for n, m in enumerate(a):
            machine = machine.replace({m: n})

X_train, X_test, y_train, y_test = train_test_split(machine.loc[:,:'waterpoint_type'],
                                                    machine.loc[:,'function'],            
                                                    random_state=0)

knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
print(f"knn: {(knn.score(X_test, y_test))}")

tree = DecisionTreeClassifier(random_state=0) 
tree.fit(X_train, y_train) 
print(f"tree: {tree.score(X_test, y_test)}")

mlp = MLPClassifier(solver='lbfgs', random_state=3).fit(X_train,y_train)
print(f"ANN: {mlp.score(X_test,y_test)}")

clf=RandomForestClassifier(n_estimators=250)
clf.fit(X_train,y_train)
print(f"rf: {clf.score(X_test,y_test)}")

