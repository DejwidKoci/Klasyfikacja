from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from plot_decision_region import plot_decision_regions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_excel("hour.xlsx")
y = pd.read_excel("hour.xlsx")['season']
X = data.drop(columns = ['instant', 'dteday', 'season'])

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.2, 
                     stratify = y,
                     random_state = 0)

# standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)



# initializing the PCA transformer 
# and logistic regression estimator
pca = PCA(n_components = 2)
lr = LogisticRegression(multi_class = 'ovr',
                        random_state = 1,
                        solver = 'lbfgs')

# dimesionality reduction
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


# covariance matrix
cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# fitting the logistic regression model on the reduced dataset
lr.fit(X_train_pca, y_train)
plot_decision_regions(X_train_pca, y_train, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()


plot_decision_regions(X_test_pca, y_test, classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc = 'lower left')
plt.tight_layout()
plt.show()

pca = PCA(n_components = None)
X_train_pca = pca.fit_transform(X_train_std)
print(pca.explained_variance_ratio_)

loadings = eigen_vecs * np.sqrt(eigen_vals)
fig, ax = plt.subplots()
ax.bar(range(14), loadings[:, 0], align = 'center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(14))
ax.set_xticklabels(data.columns[3:], rotation = 90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()



sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig, ax = plt.subplots()
ax.bar(range(14), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(14))
ax.set_xticklabels(data.columns[3:], rotation = 90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.show()