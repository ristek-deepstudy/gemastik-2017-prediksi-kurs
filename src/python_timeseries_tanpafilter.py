"""
GEMASTIK 2017
TIM DEEPSTUDY UNIVERSITAS INDONESIA
Joseph Jovito | Kerenza Dexolodeo | Wisnu Pramadhitya Ramadhan

Prediksi Fluktuasi Nilai Tukar Mata Uang Melalui Konten Berita Daring

Desc:

"""

import sqlite3
import sys
import gc
import settings
from Database import Database
from Models import Boosting, Neighbors, balanced_train, split_group, mrc, transform, clean_text

try:
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.feature_extraction.text import TfidfVectorizer as TV
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.naive_bayes import MultinomialNB as NB
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import CountVectorizer as CV
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

if __name__ == '__main__':
    DB_PATH = settings.DATABASES['default']['PATH']

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT Text,Date,Clock,Sentiment From Berita ")
    result = c.fetchall()

    db = Database(DB_PATH)
    data = {}
    label = {}

    for i in tqdm(range(len(result))):
        res = result[i]
        sess = db.find_session(res[1], res[2])[0]
        if sess not in data:
            data[sess] = []
            label[sess] = res[3]
        sentence = res[0]
        new_sentence = ""
        for j in sentence:
            if j.isalpha():
                new_sentence += j
            else:
                new_sentence += " "
        data[sess].append(new_sentence)

    chronology = list(data.keys())
    for i in range(len(data.keys())):
        for j in range(i + 1, len(data.keys())):
            if chronology[i] > chronology[j]:
                chronology[i], chronology[j] = chronology[j], chronology[i]

    date = [chronology[len(chronology) * i // 6-1] for i in range(1,7)]

    del data, label
    gc.collect()

    clf_option = [
        Boosting(),
        LR(n_jobs = -1),
        NB(),
        LinearSVC(),
        Neighbors(),
        RFC()
    ]

    mre_pred = []

    for iter in tqdm(range(5)):
        if settings.DEBUG_MODE:
            print("Memulai pengambilan data")

        mre_total = []
        query = "Select * from berita WHERE Date <= "+str(date[iter])
        c.execute(query)
        train_data = c.fetchall()

        query = "Select * from berita WHERE Date <= "+str(date[iter+1])+" AND "+str(date[iter])
        c.execute(query)
        testData = c.fetchall()

        # Do cross-validation to choose the best feature selection
        X = clean_text([i[3] for i in train_data])
        y = [i[7] for i in train_data]

        # @TODO Mengurangi memori

        if settings.DEBUG_MODE:
            print("Memulai CV")

        group = split_group(5, X=X, y=y)
        X_kFold = [[X[j] for j in k] for k in group]
        y_kFold = [[y[j] for j in k] for k in group]
        group.clean()
        counter_list = []
        select_list = []
        mre_total = []

        for i in range(5):
            X_train = []
            y_train = []

            X_test = []
            y_test = []

            for j in range(5):
                if j != i:
                    X_train.extend(X_kFold[j])
                    y_train.extend(y_kFold[j])
                else:
                    testIndex = j

            X_train = transform(X_train)
            X_test = transform(X_kFold[testIndex])

            assert len(X_train) == len(y_train)
            X_train_new , y_train_new = balanced_train(X_train, y_train, 'CV')
            X_train.clear()
            y_train.clear()
            counter_list.append(CV(ngram_range=(2, 2), min_df=5))
            train_vector = counter_list[-1].fit_transform(X_train_new)
            test_vector  = counter_list[-1].transform(X_test)

            select_list.append(SelectKBest(chi2, k = min(10000, train_vector.shape[1])))

            train_vector = select_list[-1].fit_transform(train_vector, y_train_new)
            test_vector  = select_list[-1].transform(test_vector)

            mre_total.append(0)
            for j in clf_option:
                j.fit(train_vector, y_train_new)
                prediction = j.predict(test_vector)
                mre_total[-1] += mrc(prediction, y_kFold[testIndex])

        index = mre_total.index(max(mre_total))
        select_counter = counter_list[index]
        select_feature = select_list[index]

        counter_list.clear()
        select_list.clear()

        mre_pred.append({'post':[],
                        'chronological':[]})

        X_train_new = []
        y_train_new = []
        # Generating boosting data
        for i in range(5):
            X_temp , y_temp = balanced_train(X, y, 'Boosting')
            X_train_new.append(X_temp)
            y_train_new.append(y_temp)

        X.clear()
        y.clear()

        train_vector = [select_counter.transform(I) for I in X_train_new]
        X_train_new.clear()
        train_vector = [select_list.transform(I) for I in train_vector]

        # Create the test set of chronological entries
        length_of_test_data = {}
        test_X = []
        data_y = {}

        for entry in testData:
            if db.find_session(entry[5], entry[6])[0] not in length_of_test_data:
                length_of_test_data[db.find_session(entry[5], entry[6])[0]] = 0
                data_y[db.find_session(entry[5], entry[6])[0]] = entry[7]
            length_of_test_data[db.find_session(entry[5], entry[6])[0]] += 1
            test_X.append(entry[3])

        test_X = select_counter.transform(transform(test_X))
        test_X.clean()
        test_vector = select_list.transform(test_X)

        if settings.DEBUG_MODE:
            print("Mulai training")

        for opt in clf_option:
            post_predict = array([int(0) for J in range(test_vector.shape[0])])
            for boosting_iter in range(5):
                opt.fit(train_vector[boosting_iter], y_train_new[boosting_iter])
                post_predict += opt.predict(test_vector)

            post_predict = [[-1, 1][jj > 0] for jj in post_predict]

            day_predict = []
            post_y = []
            day_y = []
            sum_index = 0

            for date_tested in length_of_test_data:
                day_predict.append([-1, 1][sum(post_predict[sum_index:sum_index + length_of_test_data[date_tested]]) > 0])
                post_y.extend([data_y[date_tested] for I in range(length_of_test_data[date_tested])])
                day_y.append(data_y[date_tested])
                sum_index += length_of_test_data[date_tested]

            mre_pred[-1]['post'].append(mrc(post_predict, post_y))
            mre_pred[-1]['chronological'].append(mrc(day_predict, day_y))
        test_vector.clean()
        day_predict.clean()
        post_y.clean()
        day_y.clean()

    for i in range(len(settings.ALGORITHM)):
        chronological = [j['chronological'][i] for j in mre_pred]
        post = [j['post'][i] for j in mre_pred]
        print("%s -> (chronological) %f (post) %f " %
              (settings.ALGORITHM[i], sum(chronological) / 5, sum(post) / 5))

    if settings.DEBUG_MODE:
        print(mre_pred)