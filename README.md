# 📌 TTC4900 News Text Classification Project 🚀✨🎯

This project implements a text classification application using the TTC4900 News dataset. The objective is to explore various **Natural Language Processing (NLP)** techniques—from **traditional machine learning methods** to **complex deep learning architectures** and **transformer-based models**—to classify **Turkish news articles** effectively. 🎭📊📚

---

## 📜 Table of Contents 📚🔍📌
- [Objective](#objective)
- [Findings & Insights](#findings--insights)
- [Overview](#overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Extraction](#feature-extraction)
- [Models Implemented](#models-implemented)
  - [Traditional Machine Learning Models](#traditional-machine-learning-models)
  - [Artificial Neural Networks (ANN)](#artificial-neural-networks-ann)
  - [Complex Neural Network Architectures](#complex-neural-network-architectures)
  - [Transformer-Based Models](#transformer-based-models)
- [Models Performance and Analysis](#models-performance-and-analysis)
- [Conclusion](#conclusion)
- [References](#references)

---

## 🎯 Objective 🏆💡
The goal of this project is to classify texts from the **TTC4900 News Dataset**, which consists of **4,900 samples**. The workflow involves:

![download](https://github.com/user-attachments/assets/5ebece87-e887-47d5-a01b-3910ce7c3fb9)

- **Data Preprocessing** (Tokenization & Lemmatization)
- **Feature Extraction** (BoW, TF-IDF, Word2Vec)
- **Machine Learning Algorithms** (LightGBM, XGBoost, SVM, Naive Bayes)
- **Deep Learning Models** (CNN, BiLSTM)
- **Transformer-Based Models** (Fine-tuned BERT)
- **Performance Evaluation** using **Accuracy, F1 Score, Precision, Recall, Confusion Matrix, and ROC-AUC graphs** 📊📈🎯

---

## 🔍 Findings & Insights 💡📌📊

### 📢 ComplexANN Performance 🚀📉📈
- **ComplexANN models** were trained using both **randomly initialized embeddings** and **trmodel fine-tune embeddings**.
- The use of **trmodel fine-tune embeddings** significantly improved performance:
  - ✅ **Accuracy with random embeddings:** 72.14%
  - ✅ **Accuracy with trmodel fine-tune embeddings:** 72.24%
- **Reason:** Leveraging **semantic context from pre-trained models** improved text understanding.
- **ComplexANN used an embedding matrix of 400 dimensions.** 🧩🔢🔍

### 🚀 Transformer-Based Model Performance 🌍🔎📢
- Transformer-based models were evaluated using:
  - **Language-specific models**, such as **[savasv/bert-turkish-text-classification](https://huggingface.co/savasy/bert-turkish-text-classification)**.
  - **Multilingual BERT models** fine-tuned for Turkish text classification.
- **Best accuracy achieved: 95.81%**, demonstrating the **effectiveness of transformer architectures** for Turkish text classification. 🏆📈📊

---

## 🛠 Data Preprocessing 🔄🧹📝

1. **Tokenization**: Splitting the text into smaller units (e.g., words or sentences).  
2. **Lemmatization**: Converting words to their root forms for normalization.  
3. **Building Vocabulary**: Creating a vocabulary from the dataset.  
4. **Converting to Numerical Representations**: Transforming text into numeric formats suitable for model input.  
5. **Padding**: Standardizing input size to **100 tokens per text** to ensure uniform sequence lengths.  

---

## 🔍 Feature Extraction Techniques 🧠🛠📈

1. **Bag of Words (BoW)**: A simple feature extraction method based on word frequency.  
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: A more advanced method that considers word importance in a document relative to the entire corpus.  
3. **Word2Vec**: Converts words into **500-dimensional vectors** to capture semantic information.  
4. **trmodel Fine-tune embeddings with the dataset** 🏆🎯🔍

---

## 🚀 Model Training 🎭🤖⚡

### 🏆 Traditional Machine Learning Models 🎲🎯📊
- **LightGBM**
- **XGBoost (XGBM)**
- **Naive Bayes**
- **SVM-RBF**

### 🤖 Neural Networks 🧠💡🔢
- **SimpleANN**: A basic artificial neural network.
- **ComplexANN**: An advanced neural network model trained using:
  - **Randomly initialized embeddings**
  - **trmodel fine-tune embeddings**, which utilized a **pre-trained embedding matrix of 400 dimensions**.

### 🔥 Transformer-Based Models 🚀📢📚
- **[savasv/bert-turkish-text-classification](https://huggingface.co/savasy/bert-turkish-text-classification)**  
  - **Accuracy: 95.81%** 🎯📊🏆
- **Bert-tr**  
  - **Accuracy: 92.55%** 📌✅📊
- **Bert-multilingual**  
  - **Accuracy: 91.73%** 🌍✅📈

---
## Models Performance and Analysis

The following table summarizes the performance of each model based on different vectorization methods:

| Model             | Vectorization Method | Accuracy    | Precision   | Recall      | F1 Score    |
|-------------------|----------------------|-------------|-------------|-------------|-------------|
| LightGBM          | Bag Of Words         | 0.781632653 | 0.780277734 | 0.781632653 | 0.779840816 |
| Xgbm              | Bag Of Words         | 0.833673469 | 0.834370001 | 0.833673469 | 0.833633860 |
| Naive Bayes       | Bag Of Words         | 0.712244898 | 0.728540108 | 0.712244898 | 0.706043581 |
| SVM-RBF           | Bag Of Words         | 0.808163265 | 0.808804112 | 0.808163265 | 0.807711153 |
| SimpleANN         | Bag Of Words         | 0.835714286 | 0.838245168 | 0.835714286 | 0.836044774 |
| LightGBM          | TF_IDF               | 0.777551020 | 0.776787810 | 0.777551020 | 0.775896840 |
| Xgbm              | TF_IDF               | 0.825510204 | 0.827294439 | 0.825510204 | 0.825669060 |
| Naive Bayes       | TF_IDF               | 0.773469388 | 0.776263559 | 0.773469388 | 0.769042589 |
| SVM-RBF           | TF_IDF               | 0.856122449 | 0.857022604 | 0.856122449 | 0.856407709 |
| SimpleANN         | TF_IDF               | 0.829591837 | 0.832747304 | 0.829591837 | 0.830562589 |
| LightGBM          | Word2Vec             | 0.674489796 | 0.677474906 | 0.674489796 | 0.674571800 |
| Xgbm              | Word2Vec             | 0.711224490 | 0.711188704 | 0.711224490 | 0.710625806 |
| Naive Bayes       | Word2Vec             | 0.495918367 | 0.514999420 | 0.495918367 | 0.489125342 |
| SVM-RBF           | Word2Vec             | 0.586734694 | 0.589540660 | 0.586734694 | 0.582359467 |
| SimpleANN         | Word2Vec             | 0.605102041 | 0.621591032 | 0.605102041 | 0.603533385 |
| ComplexANN        | padding              | 0.721428571 | 0.733368195 | 0.721428571 | 0.720124251 |
| ComplexANN        | Finetunetrmodel      | 0.722448980 | 0.739737884 | 0.722448980 | 0.724979968 |
| Bert-savasy       | text-class           | 0.958163265 | 0.958747343 | 0.958163265 | 0.958115651 |
| Bert-tr           | finetune             | 0.925510204 | 0.925940165 | 0.925510204 | 0.925348295 |
| Bert-multilingual | finetune             | 0.917346939 | 0.917939191 | 0.917346939 | 0.916979723 |

### Analysis

- **BERT Models Dominate:**  
  The **Bert-savasy** model achieves the highest performance across all metrics (Accuracy: 95.82%, Precision: 95.87%, Recall: 95.82%, F1 Score: 95.81%). This confirms that transformer-based models, when properly fine-tuned for Turkish text classification, can significantly outperform both traditional and simpler neural network models.

- **Traditional Models with TF-IDF:**  
  Among traditional machine learning algorithms, SVM-RBF using TF-IDF yields excellent performance (Accuracy: 85.61%), highlighting the effectiveness of TF-IDF in capturing relevant text features.

- **Word2Vec Limitations:**  
  The results indicate that models using **Word2Vec** vectorization generally underperform compared to those using BoW or TF-IDF representations. This may suggest that, for this particular dataset and task, simpler feature extraction methods capture the necessary information more effectively.

- **ANN and Complex Architectures:**  
  The simple ANN models achieve competitive results with traditional classifiers, while the more complex neural network (ComplexANN) shows moderate performance. This suggests that while deep architectures have potential, their benefits may require further tuning or more data to fully leverage.

- **Overall Insight:**  
  The superior performance of the Bert-savasy model underscores the advantage of leveraging transformer architectures for language-specific tasks. Fine-tuning pre-trained models for the target language (Turkish, in this case) can yield state-of-the-art results, even when compared to carefully optimized traditional models.
![image](https://github.com/user-attachments/assets/96536278-a69f-4488-ad21-61683f20690b)

---
## 🎯 Conclusion 📊🔍🏆

- **Transformer-based models outperformed all other models**, with **savasv/bert-turkish-text-classification achieving 95.81% accuracy**.
- Among **non-Transformer models**, **SVM-RBF with TF-IDF** was the best, achieving **85.61% accuracy**.
- **ComplexANN showed potential** (72.24% accuracy) but was constrained by:
  - **Dataset size (4,900 samples)**
  - **Embedding matrix dimensions (400 dimensions)**
  
✅ **Future improvements** could include:
- Using **larger embedding matrices (e.g., 768 dimensions)**
- Increasing **training data size** 📈🧠💡

---

## 📚 References 📜🔗📌

- **TTC4900 News Dataset**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/savasy/ttc4900)
- **Bert Multilingual**: [Multilingual-BERT Hugging Face](https://huggingface.co/google-bert/bert-base-multilingual-cased)
- **Bert Turkish**: [Turkish-BERT Hugging Face](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- **Bert Turkish Fine-Tuned With Dataset**: [Turkish-BERT Fine-tune](https://huggingface.co/savasy/bert-turkish-text-classification)
- **Fine-tuned Model Details**: [Technical Paper](https://arxiv.org/pdf/2401.17396)
- **trmodel-Word2Vec**: [Download: Turkish Word2Vec](https://drive.google.com/drive/folders/1IBMTAGtZ4DakSCyAoA4j7Ch0Ft1aFoww)

---

