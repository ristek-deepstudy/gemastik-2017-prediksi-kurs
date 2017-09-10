"""
GEMASTIK 2017
TIM DEEPSTUDY UNIVERSITAS INDONESIA
Joseph Jovito | Kerenza Dexolodeo | Wisnu Pramadhitya Ramadhan

Prediksi Fluktuasi Nilai Tukar Mata Uang Melalui Konten Berita Daring

Desc:

"""


import random
import sys
import settings

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
    '''

    '''
    def __init__(self):
        self.clf = GB()

    def fit(self, X, y):
        '''

        :param X:
        :param y:
        :return:
        '''
        self.clf.fit(X,y)

    def predict(self, X):
        '''

        :param X:
        :return:
        '''
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m].toarray()))
        return pred


class Neighbors:
    '''

    '''
    def __init__(self):
        self.clf = KNC()

    def fit(self, X, y):
        '''

        :param X:
        :param y:
        :return:
        '''
        self.clf.fit(X,y)

    def predict(self, X):
        '''

        :param X:
        :return:
        '''
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m]))
        return pred


def balanced_train(X, y, mode):
    '''
    mode has two options:
    -> 'CV'
    then balancedTrain would suit it undersampling method for
    cross validation

    -> 'Boosting'
    then balancedTrain would suit is sampling method for
    boosting

    :param X:
    :param y:
    :param mode:
    :return:
    '''
    assert mode == 'CV' or mode == 'Boosting'

    balanced_X = []
    balanced_y = []
    index = {}

    len_X, len_y = len(X), len(y)

    assert len_X == len_y

    #Mencatat indeks mana yang positif dan mana yang negatif
    for i in range(len_X):
        if y[i] not in index:
            index[y[i]] = []
        index[y[i]].append(i)

    minimum_point = min([len(i) for i in index.values()])

    #Memastikan jumlah (+) dan (-) sama
    for i in index:
        if mode == 'CV':
            chosen = random.sample(index[i],minimum_point)
        else:
            chosen = random.choices(index[i],k=minimum_point)

        for j in chosen:
            balanced_X.append(X[j])
            balanced_y.append(y[j])

    return balanced_X, balanced_y


def split_group(kFold, X, y):
    '''


    :param kFold:
    :param X:
    :param y:
    :return:
    '''

    #Me too lazy to plug in sklearn
    assert len(X) == len(y)
    index = [int(i) for i in range(len(X))]
    random.shuffle(index)

    group = [
        index[len(index)* ii // kFold : len(index) * (ii+1) // kFold]
            for ii in range(kFold)
    ]

    return group


def mrc(pred, Y):
    '''

    :param pred:
    :param Y:
    :return:
    '''
    assert len(pred) == len(Y)
    pred = array(pred)
    Y    = array(Y)

    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(pred)):
        if pred[i] == Y[i]:
            if pred[i] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred[i] == -1:
                FP += 1
            else:
                FN += 1

    if (settings.DEBUG_MODE):
        print("TP, FP, TN, FN: %f, %f, %f, %f" % (TP, FP, TN, FN))

    try:
        return ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** (0.5)
    except:
        return 0

def transform(text):
    '''

    :param text:
    :return:
    '''
    listNews = []
    for news in text:
        listSentence = []
        oldText = news
        while True:
            newText = oldText.replace("  ", " ")
            if oldText == newText:
                break
            oldText = newText
        listSentence.extend(newText)
        splitText = newText.split(" ")
        listSentence.append(" ".join([splitText[I] for I in range(0, len(splitText), 2)]))
        listSentence.append(" ".join([splitText[I] for I in range(1, len(splitText), 2)]))
        listNews.append(" SNIP ".join(listSentence))
    return listNews


def clean_text(array):
    '''

    :param array:
    :return:
    '''
    hasil = []
    for news in array:
        newSentence = ""
        for J in news:
            if J.isalpha():
                newSentence += J
            else:
                newSentence += " "
        hasil.append(newSentence)
    return hasil