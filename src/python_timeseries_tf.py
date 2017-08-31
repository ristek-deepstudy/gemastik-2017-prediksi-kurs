# TODO: Documentation!!

from Database import Database
import sys
import sqlite3
import random

# TODO: check package, exit if module not found
try:
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as GB
    from sklearn.ensemble import RandomForestClassifier as RFC
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

try:
    from tqdm import tqdm
except(ModuleNotFoundError):
    print("Excecption: 'tqdm' package not found")
    print("Close program")
    sys.exit()
    
# TODO: add DEBUG MODE PARAMETER, if DEBUG MODE == True, print() on
# TODO: Merapihkan string berida.db menjadi static di satu file setting sehingga lebih globally
# TODO: Refactor semua file yang memanggil berita.db menjadi ke static variable di file setting
# TODO: refactor, static var, const, functions on the top, class then, and __main__
conn = sqlite3.connect('berita.db')
c = conn.cursor()
c.execute("SELECT Text,Date,Clock,Sentiment From Berita ")
result = c.fetchall()

# TODO: implement tqdm
# TODO: change: var d with db
d = Database()
data = {}
label = {}

# TODO: change: var I with something meaningful
for I in result:
    session = d.cariSesi(I[1],I[2])[0]
    if session not in data:
        data[session] = []
        label[session] = I[3]
    sentence = I[0]
    newSentence = ""

    # TODO: change: var J with something meaningful
    for J in sentence:
        if J.isalpha():
            newSentence += J
        else:
            newSentence += " "
    data[session].append(newSentence)

# Sorting the chronological order
#MeTooLazySoMeBubbleSort
# TODO: sorting bubble log N^2, change to NlogN algorithm
chronology = list(data.keys())

for I in range(len(data.keys())):
    for J in range(I+1,len(data.keys())):
        if chronology[I] > chronology[J]:
            chronology[I],chronology[J] = chronology[J],chronology[I]

# TODO: DEBUG MODE
print(len(chronology))

def balancedTrain(X,y,mode):
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
    
    # TODO: chanege: len(X) len(Y) to variable then process it
    assert len(X) == len(y)
    
    #Mencatat indeks mana yang positif dan mana yang negatif
    # TODO: change to something meaningful
    for I in range(len(X)):
        if y[I] not in index:
            index[y[I]] = []
        index[y[I]].append(I)
    
    minimumPoint = min([len(I) for I in index.values()])    
    
    #Memastikan jumlah (+) dan (-) sama
    # TODO: change to something meaningful
    for I in index:
        if mode == 'CV':
            chosen = random.sample(index[I],minimumPoint)
        else:
            chosen = random.choices(index[I],k=minimumPoint)
        # TODO: change to something meaningful
        for J in chosen:
            balancedX.append(X[J])
            balancedY.append(y[J])
    return balancedX, balancedY

# TODO: Clean this function
def splitGroup(kFold):
    #Me too lazy to plug in sklearn
    assert len(X) == len(y)
    index = [int(I) for I in range(len(X))]
    random.shuffle(index)
    group = [index[len(index)*I//kFold:len(index)*(I+1)//kFold] for I in range(kFold)]
    return group

# TODO: move class to the first line
#Custom class for GradientBoosting
class Boosting():
    def __init__(self):
        self.clf = GB()
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self,X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m].toarray()))
        return pred

#Custom class for K Nearest Neighbor
class Neighbors:
    def __init__(self):
        self.clf = KNC()
    def fit(self,X,y):
        self.clf.fit(X,y)
    def predict(self,X):
        m = int(X.shape[0] ** (0.5))
        pred = []
        for I in range(m):
            pred.extend(self.clf.predict(X[I*X.shape[0]//m:(I+1)*X.shape[0]//m]))
        return pred

# TODO: clean this lines
clfOption = [Boosting(),LR(n_jobs = -1),NB(),LinearSVC(),Neighbors(),RFC()]
mrePred = []

# TODO: clean this function
def mrc(pred,Y):
    
    pred = array(pred)
    Y    = array(Y)
    
    TP, FP , TN, FN = 0,0,0,0
    
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
    #print(TP,FP,TN,FN)
    try:
        return ((TP*TN)-(FP*FN)) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**(0.5)
    except:
        return 0

# TODO: this needs to move to some function := easy to debug
# TODO: clean this lines from some unmeaningful variable name
mrePred = []
for index in range(5):
    mreTotal = []
    trainChronology = chronology[:len(chronology)*(index+1)//6]
    testChronology  = chronology[len(chronology)*(index+1)//6:len(chronology)*(index+2)//6]
    
    # Do cross-validation to choose the best feature selection
    
    X = []
    y = []
    
    for I in trainChronology:
        X.extend(data[I])
        y.extend([label[I] for J in range(len(data[I]))])
        
    group = splitGroup(5)
    XkFold = [[X[J] for J in K]for K in group]
    YkFold = [[y[J] for J in K]for K in group]
    counterList = []
    selectList = []
    mreTotal = []

    xTrain = []
    yTrain = []
    
    xTest  = []
    yTest = []
    
    for J in range(5):
        if J == index:
            for L in XkFold[J]:
                xTest.append(L)
            yTest.extend(YkFold[J])
        else:
            for L in XkFold[J]:
                xTrain.append(L)
            yTrain.extend(YkFold[J])

    assert len(xTrain) == len(yTrain)
    xTrainNew , yTrainNew = balancedTrain(xTrain,yTrain,'CV')
    counterList.append(TV(ngram_range=(2,2),min_df=5)) 
    trainVector = counterList[-1].fit_transform(xTrainNew)
    testVector  = counterList[-1].transform(xTest)
    
    selectList.append(SelectKBest(chi2, k = 10000))
    
    trainVector = selectList[-1].fit_transform(trainVector,yTrainNew)
    testVector  = selectList[-1].transform(testVector)

    mreTotal.append(0)
    for J in clfOption:
        J.fit(trainVector,yTrainNew)
        prediction = J.predict(testVector)
        mreTotal[-1] += mrc(prediction,yTest)
    
    index = mreTotal.index(max(mreTotal))
    
    mrePred.append({'post':[],
                    'chronological':[]})
    
    xTrainNew = []
    yTrainNew = []
    # Generating boosting data
    for I in range(9):
        X_temp , y_temp = balancedTrain(xTrain,yTrain,'Boosting')
        xTrainNew.append(X_temp)
        yTrainNew.append(y_temp)

    trainVector = [counterList[index].transform(I) for I in xTrainNew]
    testVector  = counterList[index].transform(xTest)
        
    trainVector = [selectList[index].transform(I) for I in trainVector]
    testVector  = selectList[index].transform(testVector)
    
    for I in clfOption:
        predict = array([int(0) for J in yTest])
        for boostingIter in range(9):
            I.fit(trainVector[boostingIter],yTrainNew[boostingIter])
            predict += I.predict(testVector)
        postPredict = [[-1,1][J>0] for J in predict]
        
        dayPredict = []
        dayY = []
        sumIndex = 0
        
        for J in testChronology:
            dayPredict.append([-1,1][sum(postPredict[sumIndex:sumIndex+len(data[J])])>0])
            dayY.append(label[J])
            sumIndex += len(data[J])
            
        mrePred[-1]['post'].append(mrc(postPredict,yTest))
        mrePred[-1]['chronological'].append(mrc(dayPredict,dayY))


# TODO: Change var name to something meaningful and move to the top and make it static var
name = ["Gradient Boosting","Logistic Regression","Naive Bayes","Linear SVC","K nearest neighbor","Random forest"]

# TODO: move this loop to single function so easy to debug and read
for I in range(len(name)):
    chronological = [J['chronological'][I] for J in mrePred]
    post = [J['post'][I] for J in mrePred]
    print("%s -> (chronological) %f (post) %f " % (name[I],sum(chronological)/5,sum(post)/5))