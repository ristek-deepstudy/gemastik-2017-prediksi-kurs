# TODO: Documentation!!

import random
import sqlite3
import sys
import settings
from Database import Database
from Models import Boosting, Neighbors

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
    
# =========================================================

# TODO: add DEBUG MODE PARAMETER, if DEBUG MODE == True, print() on

def balanced_train(X, y, mode):
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

    len_X, len_y = len(X), len(y)
    
    # TODO: chanege: len(X) len(Y) to variable then process it
    assert len_X == len_y

    # TODO: can be a function
    #Mencatat indeks mana yang positif dan mana yang negatif
    for i in range(len_X):
        if y[i] not in index:
            index[y[i]] = []
        index[y[i]].append(i)
    
    minimum_point = min([len(i) for i in index.values()])

    #TODO: can be a function
    #Memastikan jumlah (+) dan (-) sama
    for i in index:
        if mode == 'CV':
            chosen = random.sample(index[i],minimum_point)
        else:
            chosen = random.choices(index[i],k=minimum_point)

        for j in chosen:
            balancedX.append(X[j])
            balancedY.append(y[j])

    return balancedX, balancedY

# TODO: Clean this function
def split_group(kFold, X, y):
    #Me too lazy to plug in sklearn
    assert len(X) == len(y)
    index = [int(I) for I in range(len(X))]
    random.shuffle(index)
    group = [index[len(index)*I//kFold:len(index)*(I+1)//kFold] for I in range(kFold)]
    return group

# TODO: clean this function
def mrc(pred, Y):
    
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



def main():

    # path for db in settings.py
    DB_PATH = settings.DATABASES['default']['PATH']

    # create connection
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT Text,Date,Clock,Sentiment From Berita ")
    result = c.fetchall()

    # TODO: change: var d with db
    db = Database(DB_PATH)
    data = {}
    label = {}

    # TODO: implement tqdm
    # TODO: change: var I with something meaningful
    # Load data from db after fetch
    for i in tqdm(range(len(result))):
        res = result[i]
        sess = db.find_session(res[1], res[2])[0]

        if sess not in data:
            data[sess] = []
            label[sess] = res[3]
        
        sentence = res[0]
        new_sentence = ""

        for word in sentence:
            if word.isalpha():
                new_sentence += word
            else:
                new_sentence += " "

        data[sess].append(new_sentence)

    # Sorting the chronological order
    chronology = list(data.keys())
    chronology.sort()

    if settings.DEBUG_MODE == True:
        print("|Chronology|: ", len(chronology))

    clf_option = [
        Boosting(),
        LR(n_jobs = -1),
        NB(),
        LinearSVC(),
        Neighbors(),
        RFC()]

    mre_pred = []

    for index in range(5):

        train_chronology = chronology[:len(chronology)*(index+1)//6]
        test_chronology  = chronology[len(chronology)*(index+1)//6:len(chronology)*(index+2)//6]
        
        # Do cross-validation to choose the best feature selection
        X = []
        y = []
        
        for i in train_chronology:
            X.extend(data[i])
            y.extend([label[i] for j in range(len(data[i]))])
            
        group = split_group(5)

        X_Kfold = [[X[j] for j in k]for k in group]
        y_Kfold = [[y[j] for j in k]for k in group]

        counter_list = []
        select_list = []
        mre_total = []

        X_train = []
        y_train = []
        
        X_test  = []
        y_test = []
        
        for j in range(5):
            if j == index:
                for l in X_Kfold[j]:
                    X_test.append(l)

                y_test.extend(y_Kfold[l])

            else:
                for l in X_Kfold[j]:
                    X_train.append(l)

                y_train.extend(y_Kfold[j])

        assert len(X_train) == len(y_train)

        X_train_new, y_train_new = balanced_train(X_train, y_train, 'CV')

        counter_list.append(
            TV(ngram_range=(2, 2), min_df=5)
        )
        train_vector = counter_list[-1].fit_transform(X_train_new)
        test_vector  = counter_list[-1].transform(X_test)
        
        select_list.append(
            SelectKBest(chi2, k = 10000)
        )
        train_vector = select_list[-1].fit_transform(train_vector, y_train_new)
        test_vector  = select_list[-1].transform(test_vector)

        mre_total.append(0)

        for opt in clf_option:
            opt.fit(train_vector, y_train_new)
            #
            prediction = opt.predict(test_vector)
            #
            mre_total[-1] += mrc(prediction,y_test)
        
        index = mre_total.index(max(mre_total))
        
        mre_pred.append({'post':[],
                        'chronological':[]})
        
        X_train_new, y_train_new = [],[]

        # Generating boosting data
        for i in range(9):
            X_temp , y_temp = balanced_train(X_train, y_train, 'Boosting')

            X_train_new.append(X_temp)
            y_train_new.append(y_temp)

        train_vector = [counter_list[index].transform(I) for I in X_train_new]
        test_vector  = counter_list[index].transform(X_test)
            
        train_vector = [select_list[index].transform(I) for I in train_vector]
        test_vector  = select_list[index].transform(test_vector)
        
        for opt in clf_option:
            predict = array([int() for j in y_test])

            BOOSTING_ITER = 9
            for jj in range(BOOSTING_ITER):
                opt.fit(train_vector[jj],y_train_new[jj])
                predict += opt.predict(test_vector)

            post_predict = [[-1, 1][j > 0] for j in predict]

            day_predict = []
            day_y = []
            sum_index = 0

            for j in test_chronology:
                day_predict.append([-1, 1][sum(post_predict[sum_index : sum_index + len( data[j] )]) > 0])
                day_y.append(label[j])
                sum_index += len(data[j])
                
            mre_pred[-1]['post'].append(mrc(post_predict,y_test))
            mre_pred[-1]['chronological'].append(mrc(day_predict,day_y))


    # TODO: Change var name to something meaningful and move to the top and make it static var
    name = ["Gradient Boosting","Logistic Regression","Naive Bayes","Linear SVC","K nearest neighbor","Random forest"]

    # TODO: move this loop to single function so easy to debug and read
    for i in range(len(name)):
        chronological = [j['chronological'][i] for j in mre_pred]
        post = [j['post'][i] for j in mre_pred]

        print("%s -> (chronological) %f (post) %f " % (name[I],sum(chronological)/5,sum(post)/5))

if "__main__" == __name__:
    main()