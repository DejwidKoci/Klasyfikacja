# Data Processing
import pandas as pd
import numpy as np

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
from IPython.display import display
from sklearn.neighbors import KNeighborsClassifier

day = pd.read_excel("day.xlsx")

y = pd.read_excel("day.xlsx")['season']
X = day.drop(columns = ['instant', 'dteday', 'season'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

rf = RandomForestClassifier(n_estimators = 500)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    graph.view()
    



## random search

param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print('Best hyperparameters:',  rand_search.best_params_)


## Matrix

# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix = cm).plot()

knn = KNeighborsClassifier(n_neighbors=5)  # Ustawienie liczby sąsiadów
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'micro')
recall = recall_score(y_test, y_pred,  average = 'micro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)


# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()




# # Generowanie losowych próbek
# liczba_probek = 5  # Możesz zmienić liczbę próbek wedle potrzeb
# losowe_probki = hour.sample(n = liczba_probek)

# # Wyświetlenie wygenerowanych losowo próbek
# print("\nWygenerowane losowe próbki:")
# print(losowe_probki)