# Email Spam Detection Using NLP & Machine Learning

## Project Overview
This project is an **Email Spam Classifier** built as part of my learning journey in **Natural Language Processing (NLP)** and its associated libraries. The goal was to classify emails as **Spam** or **Ham** using machine learning models like **Random Forest** and **Naive Bayes**.

While the project demonstrates good predictive performance, the model appears to be **biased towards predicting emails as spam**. This could be due to **class imbalance** in the dataset or the relatively small dataset size (**5,172 emails**).

## Dataset
- The dataset consists of **5,172 emails** with **3,672 non-spam (ham)** and **1,500 spam** emails.
- Each email is represented by **3,000 numerical features**, corresponding to word frequencies.
- The final column, `Prediction`, contains the target labels: `0` (ham) and `1` (spam).

## NLP Techniques Used
- **Text Preprocessing**: Stopword removal, lemmatization, and punctuation removal using **NLTK**.
- **Feature Engineering**: TF-IDF and word frequency-based vectorization.
- **Dimensionality Reduction**: PCA & t-SNE visualizations for better feature understanding.

## Models Implemented
| Model              | Accuracy  | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
|-------------------|-----------|-----------------|--------------|----------------|
| **Random Forest** | **97.29%** | 94%             | 97%          | 95%            |
| **Naive Bayes**   | 94.78%    | 88%             | 95%          | 91%            |

- The **Random Forest model** achieved the best accuracy (**97.29%**).
- However, the **Naive Bayes model** performed better in handling spam recall.

##  Issues & Potential Improvements
1. **Class Imbalance**: The dataset is imbalanced (more ham emails). Using techniques like **SMOTE** or **weighted loss functions** may improve fairness.
2. **Small Dataset**: The model could improve with a **larger, more diverse dataset**.
3. **Alternative Models**: Trying **SVM (Support Vector Machine)** or **LSTMs** for text classification may yield better results.
4. **Threshold Tuning**: Adjusting the spam classification probability threshold might reduce false positives.

## 🛠️ How to Run the Project
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the Model**:
   ```bash
   python train.py
   ```
3. **Run the Streamlit App**:
   ```bash
   streamlit run spam.py
   ```

## 📂 Files in This Repository
- `spam.ipynb` - Jupyter notebook for training, evaluation and visualization.
- `emails.csv` - The dataset used for training and testing.
- `spam.py` - python file for training, evaluation and visualization.  
- `app.py` - Streamlit-based UI for classifying emails.

## 🔗 Future Work
- Improve class balance using resampling techniques.
- Implement additional deep learning-based NLP models.
- Fine-tune hyperparameters for better generalization.


