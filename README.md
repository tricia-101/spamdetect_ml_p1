# spamdetect_ml_p1

# 📨 AI Email Spam Detector using Deep Learning

Welcome to my Email Spam Detector project! I am exploring Machine Learning and Deep Learning to improve my data science skills. I took a foundational spam detection workflow and expanded it to build an end-to-end Neural Network.

This project processes raw email text data, balances the dataset, handles NLP (Natural Language Processing) text cleaning, and uses an **LSTM (Long Short-Term Memory) Neural Network** to accurately classify messages as either **Spam** or **Ham (Not Spam)**.

---

## 🚀 Key Features & Workflow

### 1. Data Cleaning & Preprocessing
*   **Downsampling:** Automatically balances the dataset so there is an equal amount of spam and ham emails, preventing the model from becoming biased.
*   **Punctuation Removal:** Fast text-cleaning using a C-optimized string translation map to strip out symbols (`!`, `?`, etc.).
*   **Stopword Filtering:** Uses the **NLTK** library to remove common filler words (like "the", "is", "at") that do not add predictive value.
*   **Missing Value Handling:** Converts empty data rows cleanly to empty strings to prevent code crashes during tokenization.

### 2. Data Visualization
*   Generates custom **WordClouds** using `matplotlib` to visually map the top 100 most frequent words in Spam vs. Non-Spam categories.

### 3. Text-to-Number Tokenization
*   Uses TensorFlow's `Tokenizer` to build a unique vocabulary dictionary.
*   Converts text sequences into integer arrays and standardizes their lengths to 100 words using **padding and truncating**.

### 4. Deep Learning Model Architecture
Built using **TensorFlow & Keras** with a Sequential layout:
1.  **Embedding Layer:** Learns mathematical word relationships and meanings.
2.  **LSTM Layer:** A recurrent memory layer that tracks the contextual sequence of words.
3.  **Dense Layers:** Fully-connected network layers using `relu` and `sigmoid` activation functions to calculate the final 0 (Ham) or 1 (Spam) prediction score.

---

## 🛠️ Automated Training Optimization
To keep training fast and prevent the model from memorizing data (overfitting), I integrated:
*   **EarlyStopping:** Shuts down training if test accuracy stalls for 3 consecutive cycles and reverts back to the absolute best settings.
*   **ReduceLROnPlateau:** Automatically shrinks the learning speed when progress slows down to perform precise fine-tuning.

---

## 📦 Python Libraries Used
*   `tensorflow` (Deep Learning network management)
*   `nltk` (Natural Language Toolkit for stopword dictionaries)
*   `scikit-learn` (Data splitting and downsampling)
*   `pandas` & `numpy` (Dataframe manipulation and matrix file saving)
*   `wordcloud` & `matplotlib` (Visual graphs and word clouds)

---

## 📈 Next Steps & Goals
As I continue to play around with this code, my next steps are to:
1. Try increasing the maximum word sequence length past 100 to see if it catches longer spam emails.
2. Build a custom input function where I can type a brand new sentence and have the AI guess if it is spam or ham in real-time.
