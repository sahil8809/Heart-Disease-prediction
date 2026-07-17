import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # JUST LIKE fillna [fill missing values] with  #mean, median, most_frequent, constant
from sklearn.pipeline import Pipeline    
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB  # for continuos numeric features

###
# Load Dataset
# ------------
data = pd.read_csv(r"C:\Users\mdsah\OneDrive\Desktop\PYTHON BASICS\Machine Learning\projects\heart_disease\heart_disease_cleveland.csv")
print("Shape :", data.shape)
# print(data.isnull().sum())      ## before imputation


# Handle Missing Values
imputer = SimpleImputer(strategy="most_frequent")  # REPLACE NAN values with most repeated values #mean, median, most_frequent, constant

# print(data.isnull())
# print(data.isnull().sum())
                                 
X = data.drop("target", axis=1) # remove axis 1 -> column, axis 0 -> row
y = data["target"]

## after imputation
# df = pd.DataFrame(imputer.fit_transform(X),columns=X.columns)
# print(df.isnull().sum())


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Models
models = {

    "Logistic Regression":
    Pipeline([
        ("imputer", imputer),
        ("model", LogisticRegression(max_iter=500))
    ]),

    "Decision Tree":
    Pipeline([
        ("imputer", imputer),
        ("model", DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest":
    Pipeline([
        ("imputer", imputer),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_split=5,
            random_state=42
        ))
    ]),

    "Naive Bayes":
    Pipeline([
        ("imputer", imputer),
        ("model", GaussianNB())
    ])
}

best_model = None
best_accuracy = 0

print("="*60)

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    print(f"{name} Accuracy : {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

print("="*60)

print(f"\nBest Model : {best_name}")
print(f"Best Accuracy : {best_accuracy:.4f}")

prediction = best_model.predict(X_test)

print("\nClassification Report\n")
print(classification_report(y_test, prediction))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, prediction))

# joblib.dump(best_model, "heart_model.pkl")

# print("\nheart_model.pkl Saved Successfully")

