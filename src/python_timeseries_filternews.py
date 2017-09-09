import sqlite3
import sys
import settings
from Database import Database
from Models import transform, mrc, split_group, balanced_train, Boosting, Neighbors, clean_text

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
    # TODO: change: lebih static nanti menggunakan file setting.py sehingga terpusat
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

    date = [chronology[len(chronology) * i // 6 - 1] for i in range(1, 7)]

    clf_option = [
            Boosting(),
            LR(n_jobs=-1),
            NB(), LinearSVC(),
            Neighbors(),
            RFC()
         ]
    mre_pred = []

    for iter in tqdm(range(5)):
        query = "Select * from berita WHERE Date <= " + str(date[iter]) + " AND Title LIKE '%ekono%' "
        c.execute(query)
        train_data = c.fetchall()

        query = "Select * from berita WHERE Date <= " + str(date[iter]) + " AND NOT Title LIKE '%ekono%' "
        c.execute(query)
        train_data_unknown = c.fetchall()

        query = "Select * from berita WHERE Date <= " + str(date[iter + 1]) + " AND " + str(
            date[iter]) + "< Date AND NOT Title LIKE '%ekono%' "
        c.execute(query)
        test_data = c.fetchall()

        if settings.ALGORITHM:
            print("Data berhasil difetch")
            print("Data Train Unknown: %d Data Train: %d Data Test: %d" %
                        (len(train_data_unknown), len(train_data), len(test_data)))

        filtered = []
        for i in range(0, len(train_data_unknown), len(train_data)):
            X = [j[3] for j in train_data]
            y = [int(1) for j in train_data]
            to_evaluate = [j[3] for j in train_data_unknown[i:i + len(train_data)]]
            X += to_evaluate
            y += [int(0) for j in to_evaluate]

            counter = CV()
            vector = counter.fit_transform(clean_text(X))
            to_evaluate_vector = counter.transform(clean_text(to_evaluate))

            bayes = NB()
            bayes.fit(vector, y)
            predict = bayes.predict_proba(to_evaluate_vector)

            for j in range(len(predict)):
                if predict[j][1] > 0.9:
                    filtered.append(train_data_unknown[i + j])

        if settings.DEBUG_MODE:
            print("Data berhasil difilter")
            print("Data Filtered: ", len(filtered))

        train_data += filtered

        # Do cross-validation to choose the best feature selection
        X = []
        y = []

        for i in train_data:
            X.append(i[3])
            y.append(i[7])

        group = split_group(5, X=X, y=y)
        X_kFold = [[X[j] for j in k] for k in group]
        y_kFold = [[y[j] for j in k] for k in group]
        counter_list = []
        select_list = []
        mre_total = []

        for i in range(5):
            X_train = []
            y_train = []

            X_test = []
            y_test = []

            for j in range(5):
                if j == i:
                    for l in X_kFold[j]:
                        X_test.append(l)
                    y_test.extend(y_kFold[j])

                else:
                    for l in X_kFold[j]:
                        X_train.append(l)
                    y_train.extend(y_kFold[j])

            X_train = transform(X_train)
            X_test = transform(X_test)

            assert len(X_train) == len(y_train)

            X_train_new, y_train_new = balanced_train(X_train, y_train, 'CV')
            counter_list.append(CV(ngram_range=(2, 2), min_df=5))
            train_vector = counter_list[-1].fit_transform(X_train_new)
            test_vector = counter_list[-1].transform(X_test)

            select_list.append(SelectKBest(chi2, k=min(10000, train_vector.shape[1])))

            train_vector = select_list[-1].fit_transform(train_vector, y_train_new)
            test_vector = select_list[-1].transform(test_vector)

            mre_total.append(0)
            for j in clf_option:
                j.fit(train_vector, y_train_new)
                prediction = j.predict(test_vector)
                mre_total[-1] += mrc(prediction, y_test)

        index = mre_total.index(max(mre_total))

        mre_pred.append({'post': [],
                        'chronological': []})

        X_train_new = []
        y_train_new = []
        # Generating boosting data
        for i in range(5):
            #TODO: BUGS = is X_train, y_train should be X, y???
            X_temp, y_temp = balanced_train(X_train, y_train, 'Boosting')
            X_train_new.append(X_temp)
            y_train_new.append(y_temp)

        train_vector = [counter_list[index].transform(i) for i in X_train_new]
        train_vector = [select_list[index].transform(i) for i in train_vector]

        # Create the test set of chronological entries

        length_of_test_data = {}
        test_X = []
        data_y = {}

        for entry in test_data:
            if entry[5] not in length_of_test_data:
                length_of_test_data[entry[5]] = 0
                data_y[entry[5]] = entry[7]

            length_of_test_data[entry[5]] += 1
            test_X.append(entry[3])

        test_X = counter_list[index].transform(transform(test_X))
        test_vector = select_list[index].transform(test_X)

        if settings.DEBUG_MODE:
            print("Mulai training")

        for i in clf_option:
            post_predict = array([int(0) for J in range(test_vector.shape[0])])
            for boosting_iter in range(5):
                i.fit(train_vector[boosting_iter], y_train_new[boosting_iter])
                post_predict += i.predict(test_vector)
            post_predict = [[-1, 1][J > 0] for J in post_predict]

            day_predict = []
            post_y = []
            day_y = []
            sum_index = 0

            for dateTested in length_of_test_data:
                day_predict.append([-1, 1][sum(post_predict[sum_index:sum_index + length_of_test_data[dateTested]]) > 0])
                post_y.extend([data_y[dateTested] for I in range(length_of_test_data[dateTested])])
                day_y.append(data_y[dateTested])
                sum_index += length_of_test_data[dateTested]

            mre_pred[-1]['post'].append(mrc(post_predict, post_y))
            mre_pred[-1]['chronological'].append(mrc(day_predict, day_y))


    if settings.DEBUG_MODE:
        print("mre_prediction: ", mre_pred)

    for i in range(len(settings.ALGORITHM)):
        chronological = [j['chronological'][i] for j in mre_pred]
        post = [j['post'][i] for j in mre_pred]
        print("%s -> (chronological) %f (post) %f " %
              (settings.ALGORITHM[i], sum(chronological) / 5, sum(post) / 5))
>>>>>>> 76dc7a8095b1dd019d7c97fa0e67c22d216640b5
