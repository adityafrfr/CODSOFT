import pandas as pd
import numpy as np
import re
import os
import time
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class MovieGenreClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                        ngram_range=(1, 2), lowercase=True)
        self.label_encoder = LabelEncoder()
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Linear SVM (Fast)': LinearSVC(random_state=42, C=1.0, max_iter=1000)
        }
        self.best_model = None
        self.best_accuracy = 0
        
    def load_data(self, train_file, test_file, test_solution_file):
        print("Loading data files...")
        
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ::: ')
                if len(parts) >= 4:
                    train_data.append({
                        'id': parts[0],
                        'title': parts[1],
                        'genre': parts[2],
                        'description': parts[3]
                    })
        
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ::: ')
                if len(parts) >= 3:
                    test_data.append({
                        'id': parts[0],
                        'title': parts[1],
                        'description': parts[2]
                    })
        
        test_solutions = []
        with open(test_solution_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(' ::: ')
                if len(parts) >= 4:
                    test_solutions.append({
                        'id': parts[0],
                        'title': parts[1],
                        'genre': parts[2],
                        'description': parts[3]
                    })
        
        self.train_df = pd.DataFrame(train_data)
        self.test_df = pd.DataFrame(test_data)
        self.test_solution_df = pd.DataFrame(test_solutions)
        
        print(f"Loaded {len(self.train_df)} training samples")
        print(f"Loaded {len(self.test_df)} test samples")
        print(f"Loaded {len(self.test_solution_df)} test solutions")
        
        return self.train_df, self.test_df, self.test_solution_df
    
    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text
    
    def analyze_data(self):
        print("\n=== Data Analysis ===")
        
        genre_counts = self.train_df['genre'].value_counts()
        print(f"\nGenre distribution in training data:")
        print(genre_counts)
        
        plt.figure(figsize=(12, 6))
        genre_counts.plot(kind='bar')
        plt.title('Genre Distribution in Training Data')
        plt.xlabel('Genre')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('genre_distribution.png')
        plt.show()
        
        self.train_df['description_length'] = self.train_df['description'].str.len()
        print(f"\nDescription length statistics:")
        print(self.train_df['description_length'].describe())
        
        return genre_counts
    
    def prepare_features(self):
        print("\n=== Feature Preparation ===")
        
        self.train_df['combined_text'] = (
            self.train_df['title'].fillna('') + ' ' + 
            self.train_df['description'].fillna('')
        )
        
        self.test_df['combined_text'] = (
            self.test_df['title'].fillna('') + ' ' + 
            self.test_df['description'].fillna('')
        )
        
        self.test_solution_df['combined_text'] = (
            self.test_solution_df['title'].fillna('') + ' ' + 
            self.test_solution_df['description'].fillna('')
        )
        
        self.train_df['processed_text'] = self.train_df['combined_text'].apply(self.preprocess_text)
        self.test_df['processed_text'] = self.test_df['combined_text'].apply(self.preprocess_text)
        self.test_solution_df['processed_text'] = self.test_solution_df['combined_text'].apply(self.preprocess_text)
        
        print("Creating TF-IDF features...")
        X_train_text = self.vectorizer.fit_transform(self.train_df['processed_text'])
        X_test_text = self.vectorizer.transform(self.test_df['processed_text'])
        
        y_train = self.label_encoder.fit_transform(self.train_df['genre'])
        y_test_true = self.label_encoder.transform(self.test_solution_df['genre'])
        
        print(f"Feature matrix shape: {X_train_text.shape}")
        print(f"Number of unique genres: {len(self.label_encoder.classes_)}")
        print(f"Genres: {list(self.label_encoder.classes_)}")
        
        return X_train_text, X_test_text, y_train, y_test_true
    
    def train_models(self, X_train, y_train):
        print("\n=== Model Training ===")
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            start_time = time.time()
            
            model.fit(X_train_split, y_train_split)
            
            training_time = time.time() - start_time
            print(f"{name} training time: {training_time:.2f} seconds")
            
            val_predictions = model.predict(X_val_split)
            accuracy = accuracy_score(y_val_split, val_predictions)
            model_scores[name] = accuracy
            
            print(f"{name} validation accuracy: {accuracy:.4f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model = model
                self.best_model_name = name
        
        print(f"\nBest model: {self.best_model_name} with accuracy: {self.best_accuracy:.4f}")
        
        print(f"Retraining {self.best_model_name} on full training data...")
        self.best_model.fit(X_train, y_train)
        
        return model_scores
    
    def evaluate_model(self, X_test, y_test_true):
        print("\n=== Model Evaluation ===")
        
        y_pred = self.best_model.predict(X_test)
        
        test_accuracy = accuracy_score(y_test_true, y_pred)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        genre_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(y_test_true, y_pred, target_names=genre_names))
        
        cm = confusion_matrix(y_test_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=genre_names, yticklabels=genre_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Genre')
        plt.ylabel('True Genre')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return test_accuracy, y_pred
    
    def predict_genres(self, X_test):
        predictions = self.best_model.predict(X_test)
        predicted_genres = self.label_encoder.inverse_transform(predictions)
        
        results_df = self.test_df.copy()
        results_df['predicted_genre'] = predicted_genres
        
        return results_df
    
    def save_results(self, results_df, filename='predictions.txt'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in results_df.iterrows():
                f.write(f"{row['id']} ::: {row['title']} ::: {row['predicted_genre']} ::: {row['description']}\n")
        
        print(f"Predictions saved to {filepath}")
    
    def save_model(self, filename='movie_genre_model.pkl'):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'best_accuracy': self.best_accuracy
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
        print(f"Saved model: {self.best_model_name}")
        print(f"Saved accuracy: {self.best_accuracy:.4f}")
    
    def load_model(self, filename='movie_genre_model.pkl'):
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['model']
        self.best_model_name = model_data['model_name']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.best_accuracy = model_data['best_accuracy']
        
        print(f"Model loaded from {filename}")
        print(f"Loaded model: {self.best_model_name}")
        print(f"Loaded accuracy: {self.best_accuracy:.4f}")

def main():
    classifier = MovieGenreClassifier()
    
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "train_data.txt")
    test_file = os.path.join(current_dir, "test_data.txt")
    test_solution_file = os.path.join(current_dir, "test_data_solution.txt")
    
    try:
        train_df, test_df, test_solution_df = classifier.load_data(
            train_file, test_file, test_solution_file
        )
        
        classifier.analyze_data()
        
        X_train, X_test, y_train, y_test_true = classifier.prepare_features()
        
        model_scores = classifier.train_models(X_train, y_train)
        
        test_accuracy, predictions = classifier.evaluate_model(X_test, y_test_true)
        
        classifier.save_model()
        
        results_df = classifier.predict_genres(X_test)
        
        classifier.save_results(results_df)
        
        print("\n=== Sample Predictions ===")
        for i in range(min(10, len(results_df))):
            row = results_df.iloc[i]
            true_genre = test_solution_df.iloc[i]['genre']
            print(f"Movie: {row['title']}")
            print(f"Predicted: {row['predicted_genre']}")
            print(f"Actual: {true_genre}")
            print(f"Correct: {'✓' if row['predicted_genre'] == true_genre else '✗'}")
            print("-" * 50)
        
        print(f"\nFinal Test Accuracy: {test_accuracy:.4f}")
        print(f"Best Model: {classifier.best_model_name}")
        
        print("\n=== Training Summary ===")
        print(f"✓ Trained {len(classifier.models)} different models")
        print(f"✓ Best model: {classifier.best_model_name}")
        print(f"✓ Validation accuracy: {classifier.best_accuracy:.4f}")
        print(f"✓ Test accuracy: {test_accuracy:.4f}")
        print(f"✓ Model trained on {len(classifier.train_df)} training samples")
        print(f"✓ Model tested on {len(classifier.test_solution_df)} test samples")
        print(f"✓ Model saved to 'movie_genre_model.pkl'")
        print(f"✓ Predictions saved to 'predictions.txt'")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e}")
        print("Please make sure all data files are in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()