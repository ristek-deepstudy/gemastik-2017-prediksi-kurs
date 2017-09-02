try:
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GB
    from sklearn.ensemble import RandomForestClassifier as RFC, GradientBoostingClassifier as GB
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier as KNC
    from sklearn.naive_bayes import MultinomialNB as NB
except(ModuleNotFoundError):
    print("Excecption: Scikit-learn package not found")
    print("Close program")
    sys.exit()

try:
    from numpy import array
except(ModuleNotFoundError):
    print("Excecption: 'numpy' package not found")
    print("Close program")
    sys.exit()


class Boosting():
    def __init__(self):
        self.clf = GB()

    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m].toarray()))
        return pred


class Neighbors:
    def __init__(self):
        self.clf = KNC()

    def fit(self, X, y):
        self.clf.fit(X,y)

    def predict(self, X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m]))
        return pred

