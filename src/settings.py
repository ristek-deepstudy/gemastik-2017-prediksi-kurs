import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEBUG_MODE = True

TOKEN = "e0478de37e35bac1722299a5716f204843c7a0f011fd5447"

DATABASES = {
    'default': {
        'ENGINE': 'sqlite3',
        'PATH': BASE_DIR + '/db/berita.db'
    }
}

ALGORITHM = ["Gradient Boosting",
            "Logistic Regression",
            "Naive Bayes",
            "Linear SVC",
            "K nearest neighbor",
            "Random forest"]