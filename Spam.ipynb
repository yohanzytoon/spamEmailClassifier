# %% [markdown]
# # Import necessary libraries

# %%
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import pickle 
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
from wordcloud import WordCloud  # fixed import

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# %% [markdown]
# # Load and Explore the Dataset

# %%
emails_df = pd.read_csv("emails.csv")

# Display dataset shape and column info 
print("Dataset shape: ", emails_df.shape)
print(emails_df.info())

# Display class distribution in the label column
print("Class distribution:\n", emails_df['Prediction'].value_counts())

# %% [markdown]
# # Visualize Class Distribution

# %%
plt.figure(figsize=(6, 4))
sns.countplot(x='Prediction', data=emails_df, palette='coolwarm')
plt.title('Class Distribution: Spam vs Ham')
plt.xlabel('Prediction (0 = Ham, 1 = Spam)')
plt.ylabel('Count')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()

print(emails_df['Prediction'].value_counts())

# %% [markdown]
# # Analyze Feature Columns
# (Assuming each column from the 2nd to second-last represents word frequencies)

# %%
print(emails_df.describe())

# For example, visualize the distribution for the word "the" (if exists)
if 'the' in emails_df.columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(emails_df['the'], bins=30, kde=True, color='purple')
    plt.title("Distribution of the Word 'the'")
    plt.xlabel("Frequency")
    plt.ylabel("Count")
    plt.show()
else:
    print("Column 'the' not found in the dataset.")

# Plot the mean of the top 20 features grouped by class
feature_cols = emails_df.columns[1:-1]  # assuming first col may be raw text and last col is label
feature_means = emails_df[feature_cols].groupby(emails_df['Prediction']).mean()
plt.figure(figsize=(10, 6))
feature_means.T.head(20).plot(kind='bar', figsize=(15, 6), stacked=True)
plt.title('Top 20 Features: Mean Frequency (Grouped by Prediction)')
plt.ylabel('Mean Frequency')
plt.xlabel('Words (Features)')
plt.show()

# %% [markdown]
# # Correlation Heatmap

# %%
correlation_matrix = emails_df[feature_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmax=1.0, vmin=-1.0, square=True)
plt.title("Correlation Heatmap of Features")
plt.show()

# %% [markdown]
# # Dimensionality Reduction with PCA and t-SNE

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(emails_df[feature_cols])
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=emails_df['Prediction'], palette='coolwarm')
plt.title('PCA: 2D Projection of Feature Space')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Prediction', loc='best')
plt.show()

# t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(emails_df[feature_cols])
plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=emails_df['Prediction'], palette='coolwarm')
plt.title('t-SNE: 2D Projection of Feature Space')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Prediction', loc='best')
plt.show()

# %% [markdown]
# # Split Features and Labels
# (Here, we use all columns except the first (which could be raw email text) and the last (label))

# %%
X = emails_df[feature_cols]
y = emails_df["Prediction"]

print("Feature matrix shape: ", X.shape)
print("Label shape: ", y.shape)

# %% [markdown]
# # Split Data into Training and Testing Sets

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training set shape: ", X_train.shape)
print("Test set shape: ", X_test.shape)

# %% [markdown]
# # Model Training: Random Forest and Naive Bayes

# %%
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# %% [markdown]
# # Evaluate Models on Test Data

# %%
rf_preds = rf_model.predict(X_test)
nb_preds = nb_model.predict(X_test)

print("Random Forest accuracy: ", accuracy_score(y_test, rf_preds))
print("Naive Bayes accuracy: ", accuracy_score(y_test, nb_preds))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("\nNaive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_preds))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, rf_preds))
print("\nNaive Bayes Classification Report:\n", classification_report(y_test, nb_preds))

# %% [markdown]
# # Cross-Validation Scores

# %%
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
nb_cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring='accuracy')
print('Random Forest CV Scores:', rf_cv_scores)
print('Mean Accuracy (RF):', np.mean(rf_cv_scores))
print('Naive Bayes CV Scores:', nb_cv_scores)
print('Mean Accuracy (NB):', np.mean(nb_cv_scores))

# %% [markdown]
# # Hyperparameter Tuning

# %%
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

nb_param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'fit_prior': [True, False]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42),
                              rf_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
rf_grid_search.fit(X_train, y_train)

nb_grid_search = GridSearchCV(MultinomialNB(),
                              nb_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
nb_grid_search.fit(X_train, y_train)

# %% [markdown]
# # Evaluate Tuned Models

# %%
best_rf = rf_grid_search.best_estimator_
best_nb = nb_grid_search.best_estimator_

rf_tuned_preds = best_rf.predict(X_test)
rf_tuned_accuracy = accuracy_score(y_test, rf_tuned_preds)
rf_tuned_report = classification_report(y_test, rf_tuned_preds)
rf_tuned_conf_matrix = confusion_matrix(y_test, rf_tuned_preds)

print("Tuned Random Forest Metrics:")
print(f"Accuracy: {rf_tuned_accuracy}")
print("Confusion Matrix:\n", rf_tuned_conf_matrix)
print("Classification Report:\n", rf_tuned_report)

nb_tuned_preds = best_nb.predict(X_test)
nb_tuned_accuracy = accuracy_score(y_test, nb_tuned_preds)
nb_tuned_report = classification_report(y_test, nb_tuned_preds)
nb_tuned_conf_matrix = confusion_matrix(y_test, nb_tuned_preds)

print("\nTuned Naive Bayes Metrics:")
print(f"Accuracy: {nb_tuned_accuracy}")
print("Confusion Matrix:\n", nb_tuned_conf_matrix)
print("Classification Report:\n", nb_tuned_report)

# %% [markdown]
# # Compare and Select the Best Model

# %%
if rf_tuned_accuracy > nb_tuned_accuracy:
    best_model = best_rf
    best_model_name = "Random Forest"
    best_accuracy = rf_tuned_accuracy
else:
    best_model = best_nb
    best_model_name = "Naive Bayes"
    best_accuracy = nb_tuned_accuracy

print(f"\nThe best model is: {best_model_name} with an accuracy of {best_accuracy}")

# %% [markdown]
# # Save the Best Model

# %%
model_filename = f"best_{best_model_name.lower().replace(' ', '_')}_model.pkl"
pickle.dump(best_model, open(model_filename, 'wb'))
print(f"Saved the best model as: {model_filename}")

# %% [markdown]
# # Create and Save a Vectorizer for Deployment
# Since our training features are based on word counts (columns 2 to second-last),
# we create a CountVectorizer with a fixed vocabulary from these columns.
# (If you used a different preprocessing for training, update accordingly.)

# %%
vocab = list(feature_cols)
vectorizer = CountVectorizer(vocabulary=vocab)
# (No need to fit because the vocabulary is fixed.)
vectorizer_filename = "vectorizer.pkl"
pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
print(f"Saved the vectorizer as: {vectorizer_filename}")

# %% [markdown]
# # Visualize Model Performance

# %%
model_names = ["Random Forest", "Naive Bayes"]
accuracies = [rf_tuned_accuracy, nb_tuned_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(model_names, accuracies, color=["blue", "orange"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()
