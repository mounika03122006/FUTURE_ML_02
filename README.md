# FUTURE_ML_02 - Support Ticket Classification & Prioritization

A machine learning system that reads IT support tickets, classifies them into a category, and assigns a priority level (High / Medium / Low) so support teams can respond faster.

## Problem
Support teams receive thousands of tickets. Sorting them by hand is slow, and urgent issues get delayed. This project automates category sorting and priority tagging.

## Dataset
IT Service Ticket Classification Dataset (Kaggle, adisongoh): 47,837 tickets, 8 categories (Hardware, HR Support, Access, Miscellaneous, Storage, Purchase, Internal Project, Administrative rights). The text was already cleaned in the dataset.
Link: https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset

## Approach
1. Explore the data: class balance, ticket length.
2. Clean the text: lowercase, keep letters only, remove stopwords and greetings, lemmatize (NLTK).
3. Convert text to numbers with TF-IDF (unigrams and bigrams).
4. Train and compare Naive Bayes, Logistic Regression and Linear SVM (80/20 stratified split).
5. Assign priority with keyword rules, then train models to predict it.
6. Evaluate with accuracy, precision, recall, F1 and confusion matrices.

## How priority is decided
The dataset has no priority column, so priority labels were created with rules:
- High: urgent words such as urgent, asap, critical, outage, locked out, cannot access, failed.
- Medium: problem words such as error, issue, slow, expired, warning, or a category that is usually important (Access, Administrative rights, Storage).
- Low: everything else.

## Results
| Task | Best model | Accuracy | Macro F1 |
|------|-----------|----------|----------|
| Category | Linear SVM | 85.3% | 0.853 |
| Priority | Linear SVM | 91.8% | 0.917 |

- Strongest categories: Purchase and Access. Weakest: Administrative rights.
- Hardware tickets are most often confused with HR Support.
- Charts: category_distribution.png, confusion_matrix.png, priority_confusion_matrix.png

## Limitations
- Priority labels come from keyword rules, so the priority score shows how well the model learned those rules, not real-world urgency. Real labelled priorities would give a more reliable result.
- Some tickets are misclassified, for example a quotation request for monitors was predicted as Hardware instead of Purchase.
- Access, Administrative rights and Storage tickets are never Low priority under the current rules.

## Files
- ticket_classification.ipynb : full code
- category_model_results.csv, priority_model_results.csv : model comparison
- confusion_matrix.png, priority_confusion_matrix.png, category_distribution.png : charts
- requirements.txt : libraries needed

## How to run
1. Download the dataset from the Kaggle link above.
2. pip install -r requirements.txt
3. Open the notebook and change the CSV path in the second cell to your file location.
4. Run all cells.

## Tools
Python, pandas, NLTK, scikit-learn, seaborn, matplotlib, Jupyter Notebook
