import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label','message'])

# 2. Text Preprocessing
nltk.download('stopwords', quiet=True)
stop = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = [w for w in text.split() if w not in stop]
    return ' '.join(tokens)

df['clean'] = df['message'].apply(clean_text)

# 3. Feature & Target
X = df['clean']
y = df['label'].map({'ham': 0, 'spam': 1})

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.2, stratify=y, random_state=42)

# 5. TF‑IDF Vectorization
tfidf = TfidfVectorizer(min_df=2, max_df=0.9, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 6. Train Model (Multinomial Naive Bayes)
clf = MultinomialNB(alpha=0.1)
clf.fit(X_train_tfidf, y_train)

# 7. Evaluate
y_pred = clf.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['ham','spam']))

# 8. Save model + vectorizer
joblib.dump(clf, 'sms_spam_nb_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\nModels saved: sms_spam_nb_model.pkl, tfidf_vectorizer.pkl")
