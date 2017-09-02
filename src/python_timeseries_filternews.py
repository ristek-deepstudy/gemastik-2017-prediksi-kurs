# TODO: Dokumentasi
# TODO: check package
# TODO: mode debugging akan menyalakan print, DEBUG_MODE==True, print on

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GB
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from Database import Database
from numpy import array
from sklearn.naive_bayes import MultinomialNB as NB
import sqlite3
import tqdm
import random

# TODO: change: lebih static nanti menggunakan file setting.py sehingga terpusat
conn = sqlite3.connect('berita.db')
c = conn.cursor()
c.execute("SELECT Text,Date,Clock,Sentiment From Berita ")
result = c.fetchall()

# TODO: menunggu refactor database
# TODO: use meaningful name for variable
d = Database()
data = {}
label = {}
for I in result:
    session = d.cariSesi(I[1], I[2])[0]
    if session not in data:
        data[session] = []
        label[session] = I[3]
    sentence = I[0]
    newSentence = ""
    for J in sentence:
        if J.isalpha():
            newSentence += J
        else:
            newSentence += " "
    data[session].append(newSentence)

# TODO: change sorting algorithm to NlogN algorithm using new class
# Sorting the chronological order
# MeTooLazySoMeBubbleSort

chronology = list(data.keys())

for I in range(len(data.keys())):
    for J in range(I + 1, len(data.keys())):
        if chronology[I] > chronology[J]:
            chronology[I], chronology[J] = chronology[J], chronology[I]

date = [chronology[len(chronology) * I // 6 - 1] for I in range(1, 7)]


def balancedTrain(X, y, mode):
    '''
    mode has two options:
    -> 'CV'
    then balancedTrain would suit it undersampling method for
    cross validation

    -> 'Boosting'
    then balancedTrain would suit is sampling method for
    boosting
    '''
    assert mode == 'CV' or mode == 'Boosting'
    balancedX = []
    balancedY = []

    index = {}

    assert len(X) == len(y)

    # TODO: meaningful name for variable di lines function ini

    # Mencatat indeks mana yang positif dan mana yang negatif
    for I in range(len(X)):
        if y[I] not in index:
            index[y[I]] = []
        index[y[I]].append(I)

    minimumPoint = min([len(I) for I in index.values()])

    # Memastikan jumlah (+) dan (-) sama
    for I in index:
        if mode == 'CV':
            chosen = random.sample(index[I], minimumPoint)
        else:
            chosen = random.choices(index[I], k=minimumPoint)
        for J in chosen:
            balancedX.append(X[J])
            balancedY.append(y[J])
    return balancedX, balancedY


def transform(text):
    # TODO: dokumentasi
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


def splitGroup(kFold):
    # TODO: dokumentasi
    # Me too lazy to plug in sklearn
    assert len(X) == len(y)
    index = [int(I) for I in range(len(X))]
    random.shuffle(index)

    # TODO: clean this line
    group = [index[len(index) * I // kFold:len(index) * (I + 1) // kFold] for I in range(kFold)]
    return group


# Custom class for GradientBoosting
class Boosting():
    # TODO: dokumentasi
    def __init__(self):
        self.clf = GB()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I * X.shape[0] // m:(I + 1) * X.shape[0] // m].toarray()))
        return pred


# Custom class for K Nearest Neighbor
class Neighbors:
    # TODO: dokumentasi
    def __init__(self):
        self.clf = KNC()

    def fit(self, X, y):
        self.clf.fit(X, y)

    def predict(self, X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        # TODO: Clean these lines
        for I in range(m):
            pred.extend(self.clf.predict(X[I * X.shape[0] // m:(I + 1) * X.shape[0] // m]))
        return pred


def bersihkanTeksBerita(array):
    # TODO: dokumentasi dan meaningful name
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


clfOption = [Boosting(), LR(n_jobs=-1), NB(), LinearSVC(), Neighbors(), RFC()]
mrePred = []


def mrc(pred, Y):
    # TODO: dokumentasi dan meaningful name
    assert len(pred) == len(Y)
    pred = array(pred)
    Y = array(Y)

    TP, FP, TN, FN = 0, 0, 0, 0

    for I in range(len(pred)):
        if pred[I] == Y[I]:
            if pred[I] == 1:
                TP += 1
            else:
                TN += 1
        else:
            if pred[I] == -1:
                FP += 1
            else:
                FN += 1
    print(TP, FP, TN, FN)
    try:
        return ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** (0.5)
    except:
        return 0


# TODO: should del this stucture?
del result

# TODO: DEBUG MODE
# TODO: ada baiknya lines ke bawah ini pake function per tugas jadi debug lebih mudah
mrePred = []
for iter in range(5):
    mreTotal = []
    query = "Select * from berita WHERE Date <= " + str(date[iter]) + " AND Title LIKE '%ekono%' "
    c.execute(query)
    trainData = c.fetchall()

    query = "Select * from berita WHERE Date <= " + str(date[iter]) + " AND NOT Title LIKE '%ekono%' "
    c.execute(query)
    trainDataUnknown = c.fetchall()

    query = "Select * from berita WHERE Date <= " + str(date[iter + 1]) + " AND " + str(
        date[iter]) + "< Date AND NOT Title LIKE '%ekono%' "
    c.execute(query)
    testData = c.fetchall()

    print("Data berhasil difetch")
    print(len(trainDataUnknown), len(trainData), len(testData))
    filtered = []

    # TODO: meaningful names
    for I in range(0, len(trainDataUnknown), len(trainData)):
        X = [J[3] for J in trainData]
        y = [int(1) for J in trainData]
        toEvaluate = [J[3] for J in trainDataUnknown[I:I + len(trainData)]]
        X += toEvaluate
        y += [int(0) for J in toEvaluate]

        counter = CV()
        vector = counter.fit_transform(bersihkanTeksBerita(X))
        toEvaluateVector = counter.transform(bersihkanTeksBerita(toEvaluate))

        bayes = NB()
        bayes.fit(vector, y)
        predict = bayes.predict_proba(toEvaluateVector)

        for J in range(len(predict)):
            if predict[J][1] > 0.9:
                filtered.append(trainDataUnknown[I + J])

    print("Data berhasil difilter")

    trainData += filtered
    print(len(filtered))
    # Do cross-validation to choose the best feature selection

    X = []
    y = []

    for I in trainData:
        X.append(I[3])
        y.append(I[7])

    group = splitGroup(5)
    XkFold = [[X[J] for J in K] for K in group]
    YkFold = [[y[J] for J in K] for K in group]
    counterList = []
    selectList = []
    mreTotal = []

    for I in range(5):
        xTrain = []
        yTrain = []

        xTest = []
        yTest = []
        for J in range(5):
            if J == I:
                for L in XkFold[J]:
                    xTest.append(L)
                yTest.extend(YkFold[J])
            else:
                for L in XkFold[J]:
                    xTrain.append(L)
                yTrain.extend(YkFold[J])

        xTrain = transform(xTrain)
        xTest = transform(xTest)

        assert len(xTrain) == len(yTrain)
        xTrainNew, yTrainNew = balancedTrain(xTrain, yTrain, 'CV')
        counterList.append(CV(ngram_range=(2, 2), min_df=5))
        trainVector = counterList[-1].fit_transform(xTrainNew)
        testVector = counterList[-1].transform(xTest)

        selectList.append(SelectKBest(chi2, k=min(10000, trainVector.shape[1])))

        trainVector = selectList[-1].fit_transform(trainVector, yTrainNew)
        testVector = selectList[-1].transform(testVector)

        mreTotal.append(0)
        for J in clfOption:
            J.fit(trainVector, yTrainNew)
            prediction = J.predict(testVector)
            mreTotal[-1] += mrc(prediction, yTest)

    index = mreTotal.index(max(mreTotal))

    mrePred.append({'post': [],
                    'chronological': []})

    xTrainNew = []
    yTrainNew = []
    # Generating boosting data
    for I in range(5):
        X_temp, y_temp = balancedTrain(xTrain, yTrain, 'Boosting')
        xTrainNew.append(X_temp)
        yTrainNew.append(y_temp)

    trainVector = [counterList[index].transform(I) for I in xTrainNew]
    trainVector = [selectList[index].transform(I) for I in trainVector]

    # Create the test set of chronological entries

    lengthOfTestData = {}
    testX = []
    dataY = {}

    for entry in testData:
        if entry[5] not in lengthOfTestData:
            lengthOfTestData[entry[5]] = 0
            dataY[entry[5]] = entry[7]
        lengthOfTestData[entry[5]] += 1
        testX.append(entry[3])

    testX = counterList[index].transform(transform(testX))
    testVector = selectList[index].transform(testX)
    print("Mulai training")
    for I in clfOption:
        postPredict = array([int(0) for J in range(testVector.shape[0])])
        for boostingIter in range(5):
            I.fit(trainVector[boostingIter], yTrainNew[boostingIter])
            postPredict += I.predict(testVector)
        postPredict = [[-1, 1][J > 0] for J in postPredict]

        dayPredict = []
        postY = []
        dayY = []
        sumIndex = 0

        for dateTested in lengthOfTestData:
            dayPredict.append([-1, 1][sum(postPredict[sumIndex:sumIndex + lengthOfTestData[dateTested]]) > 0])
            postY.extend([dataY[dateTested] for I in range(lengthOfTestData[dateTested])])
            dayY.append(dataY[dateTested])
            sumIndex += lengthOfTestData[dateTested]

        mrePred[-1]['post'].append(mrc(postPredict, postY))
        mrePred[-1]['chronological'].append(mrc(dayPredict, dayY))
# TODO:====== till this line, ada baiknya gunakan function spesifik===========

# TODO: if __main__

name = ["Gradient Boosting", "Logistic Regression", "Naive Bayes", "Linear SVC", "K nearest neighbor", "Random forest"]
for I in range(len(name)):
    chronological = [J['chronological'][I] for J in mrePred]
    post = [J['post'][I] for J in mrePred]
    print("%s -> (chronological) %f (post) %f " % (name[I], sum(chronological) / 5, sum(post) / 5))

print(mrePred)
print(trainVector.shape[1])
print(testX[0])