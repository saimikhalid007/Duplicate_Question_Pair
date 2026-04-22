# Duplicate Question Pair Detection Model

## Introduction

This project focuses on identifying whether question pair is duplicate or not. The goal is to build a machine learning model that can understand semantic similarity between question pairs.

Duplicate question detection is important for:

* Improving search efficiency  
* Reducing redundant content  
* Enhancing user experience

We approach this as a binary classification problem, where:

* 1 → Duplicate 
* 0 → Non-duplicate

## Dataset Overview

* **id:** the id of a training set question pair
* **qid1, qid2:** unique ids of each question
* **question1, question2:** the full text of each question
* **is_duplicate:** the target variable, set to 1 if question1 and question2 have essentialy the same meaning, and 0 otherwise.

|id|qid1|qid2|question1|question2|is_duplicate|
|----|------|--------|------|------|-----------|
|339499|665522|665523|Why was Cyrus Mistry removed as the Chairman o...|Why did the Tata Sons sacked Cyrus Mistry?|1|
|289521|568878|568879|By what age would you think a man should be ma...|When my wrist is extended I feel a shock and b...	|0|
|4665|9325|9326|How would an arbitrageur seek to capitalize gi...|How would an arbitrageur seek to capitalize gi...|0|
|54203|107861|9328|Why did Quora mark my question as incomplete?|Why does Quora detect my question as an incomp...|1|
|132566|262554|91499|What is it like working with Pivotal Labs as a...|What's it like to work at Pivotal Labs?|0|

## Approach Overview

The overall pipeline followed in this project:

<img src = "EDA_Images/Screenshot%202026-04-21%20200432.png" height = 500><br>

## Data Preprocessing

Before feature extraction, the text data is cleaned to ensure consistency.

Steps performed:
* Convert text to lowercase  
* Remove leading/trailing spaces  
* Replace special characters:
  * % → percent  
  * $ → dollar  
  * ₹ → rupee  
  * @ → at  
* Handle missing/null values

This ensures the model receives standardized input.

## Feature Engineering 

We created features in multiple stages, gradually enriching the dataset.
### Basic Features
* **q1_len:** Number of characters in Question 1  
* **q2_len:**	Number of characters in Question 2  
* **q1_num_words:** Number of words in Question 1  
* **q2_num_words:** Number of words in Question 2

### Common Word Features
* **word_common:** Count of common words between both questions
* **word_total:**	Total unique words in both questions combined
* **word_share:**	Ratio of common words to total words

### Token Based Features
* **cwc_min:** This is the ratio of the number of common words to the length of the smaller question  
* **cwc_max:** This is the ratio of the number of common words to the length of the larger question  
* **csc_min:** This is the ratio of the number of common stop words to the smaller stop word count among the two questions  
* **csc_max:** This is the ratio of the number of common stop words to the larger stop word count among the two questions  
* **ctc_min:** This is the ratio of the number of common tokens to the smaller token count among the two questions  
* **ctc_min:** This is the ratio of the number of common tokens to the larger token count among the two questions  
* **last_word_eq:** If the last word in the two questions is same, 0 otherwise
* **first_word_eq:** If the first word in the two questions is same, 0 otherwise

### Length Based Features
* **mean_len:** Mean of the length of the two questions(number of words)
* **abs_len_diff:** Absolute difference between the length of the two questions(number of words)
* **longest_substr_ratio:** Ratio of the length of the longest substring among the two questions to the length of the smaller question

### Fuzzy Features
* **fuzz_ratio:** fuzz_ratio score from fuzzywuzzy
* **fuzz_partial_ratio:** fuzz_partial_ratio from fuzzywuzzy
* **token_sort_ratio:** token_sort_ratio from fuzzywuzzy
* **token_set_ratio:** token_set_ratio from fuzzywuzzy

### Bag of Words (BoW)
* Converted text into numerical vectors using CountVectorizer
* Applied separately on:
  * Question 1
  * Question 2

### Final Feature Vector

All features are combined:
* BoW (Q1)
* BoW (Q2)
* Engineered Features

> **Result:** High-dimensional feature space capturing both syntax + semantics

## Exploratory Data Analysis

<img src = "EDA_Images/Screenshot%202026-04-20%20013846.png" height = 400><br>
Questions marked as duplicates tend to share more overlapping words, indicating semantic similarity.
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20013857.png" height = 400><br>
The total number of words in question somehow differentiate duplicate and non-duplicate classes.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20013905.png" height = 400><br>
Word share (ratio of common words to total words) is a normalized metric that captures similarity more effectively than raw counts.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20014015.png" height = 400><br>
Matching first or last words provides weak signals of similarity, but cannot reliably determine duplicates on their own.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20014038.png" height = 500><br>
Duplicates consistently show higher similarity scores across all these metrics.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20014050.png" height = 500><br>
Duplicate questions tend to have similar lengths, while non-duplicates vary significantly.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20014104.png" height = 500><br>
Longer common substrings indicate structural similarity between questions.  
<br>

<img src = "EDA_Images/Screenshot%202026-04-20%20014127.png" height = 500><br>
Fuzzy matching scores effectively capture semantic similarity even when wording differs.  
<br>

> ### Overall EDA Conslusion
The exploratory data analysis reveals that similarity-based features play a crucial role in distinguishing duplicate and non-duplicate question pairs. Features such as word share, fuzzy matching scores, and longest common substring ratio show clear separation between the two classes, indicating strong predictive power. Token-based similarity features also demonstrate consistent patterns, with duplicate pairs exhibiting higher similarity values.

On the other hand, basic features such as total word count and binary indicators like first and last word equality provide limited discriminatory power. Length-based features contribute moderately, as duplicate questions tend to have similar lengths.

Overall, the analysis suggests that normalized similarity measures and fuzzy matching techniques are the most effective in identifying duplicate questions, while simple lexical features alone are insufficient.

## Model used

Random Forest and XGBoost were used as the primary models for this classification task. These tree-based ensemble methods are well-suited for handling high-dimensional data generated from Bag of Words and engineered features. Random Forest builds multiple decision trees and averages their predictions, making it robust and less prone to overfitting. XGBoost, a gradient boosting technique, improves performance by sequentially learning from previous errors and capturing complex feature interactions. Both models effectively utilize the combination of text-based and handcrafted features, resulting in strong performance for detecting duplicate questions without requiring more complex deep learning approaches.

## Conclusion

Random Forest and XGBoost perform well on this task, especially when combined with feature engineering like word overlap, token similarity, and fuzzy matching. These models capture semantic similarity effectively without deep learning. In real-world platforms, accurate duplicate detection reduces redundant content, improves search quality, and enhances user experience. Even small improvements can optimize content organization and system efficiency. This model can be used in recommendation systems to suggest similar questions and minimize repeated queries.

## Future Analysis

Future improvements can focus on better feature representation and model performance. Replacing Bag of Words with TF-IDF can improve word importance capture. Using embeddings like Word2Vec or GloVe can enhance semantic understanding. Advanced models such as LSTM or BERT can further improve contextual similarity detection. Deploying the model as a real-time API or web application can enable practical use in recommendation and search systems.

## Requirements
[Python 3.x](https://www.python.org/downloads/)  
[Jupyter Notebook](https://jupyter.org/install)  
[Scikit-learn](https://scikit-learn.org/stable/install.html)  
[Pandas](https://pandas.pydata.org/docs/getting_started/install.html)  
[NumPy](https://numpy.org/install/)  
[Seaborn](https://pypi.org/project/seaborn/)
[Matplotlib](https://matplotlib.org/stable/install/index.html)  
[FuzzyWuzzy](https://pypi.org/project/fuzzywuzzy/) 