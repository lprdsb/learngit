import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

xigua_df = pd.read_csv('xigua.csv')
xigua_df = shuffle(xigua_df, random_state=2022)
X = xigua_df[[col for col in xigua_df.columns if col != 'label_haogua']]
y = xigua_df['label_haogua']

numeric_features = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]
categorical_features = [col for col in X.columns if X[col].dtype == 'object']

# print(categorical_features)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

clf = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", SVC())])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2022)
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}

# print(X_train, y_train)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))

grid = GridSearchCV(clf, param_grid, cv=4).fit(X_train, y_train)
print("model score: %.3f" % grid.score(X_test, y_test))