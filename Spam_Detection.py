# NECESSARY LIBRARIES
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import requests
from io import BytesIO
import zipfile

# DATASET( WE HAVE SELECTED THIS WITH THE HELP OF CHAT GPT)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
r = requests.get(url)
with zipfile.ZipFile(BytesIO(r.content), 'r') as z:
    with z.open('SMSSpamCollection') as f:
        df = pd.read_csv(f, sep='\t', names=['label', 'message'])

#'ham' to 0 and 'spam' to 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

#training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

#text data into a feature matrix(Count Vectorizer)
Vectorizer = CountVectorizer()
X_train = Vectorizer.fit_transform(X_train)
X_test = Vectorizer.transform(X_test)

#Naive Bayes classifier
used_classifier = MultinomialNB()
used_classifier.fit(X_train, y_train)

y_pred = used_classifier.predict(X_test)

#PERFORMANCE OF MODEL
Accuracy = accuracy_score(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test.tolist(), y_pred.tolist())
classification_report = classification_report(y_test, y_pred)

print(f"Accuracy: {Accuracy}")
print(f"Confusion Matrix:\n{confusion_matrix}")
print(f"Classification Report:\n{classification_report}")

#EXAMPLES
# Example 1
new_email = ["Congratulations! You've won a lottery."]
new_email_transformed = Vectorizer.transform(new_email)
prediction = used_classifier.predict(new_email_transformed)

print("Predictions for new emails:",prediction)