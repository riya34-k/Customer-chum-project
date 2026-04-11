import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()
print(df.isnull().sum())
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True) * 100)
print(pd.crosstab(df["Contract"], df["Churn"]))
print(pd.crosstab(df["tenure"] < 12, df["Churn"]))
print(pd.crosstab(df["MonthlyCharges"] > df["MonthlyCharges"].mean(), df["Churn"]))
print(pd.crosstab(df["InternetService"], df["Churn"]))
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(x="Churn", data=df)
plt.show()
sns.countplot(x="Contract", hue="Churn", data=df)
plt.show()
sns.histplot(data=df, x="MonthlyCharges", hue="Churn", kde=True)
plt.show()
sns.histplot(data=df, x="tenure", hue="Churn", kde=True)
plt.show()
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.show()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])
from sklearn.model_selection import train_test_split
X = df.drop("Churn", axis=1)
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))