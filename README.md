📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom industry. Acquiring a new customer costs 5x more than retaining an existing one. This project analyses the behaviours and patterns of churned customers to help businesses take proactive retention action.

📂 Dataset

Source: Telco Customer Churn — Kaggle
Size: 7,043 rows × 21 columns
Target Variable: Churn (Yes / No)
Features include: Contract type, tenure, monthly charges, internet service, payment method, and more.

🛠️ Tools & Libraries
Category  Tools
Language  Python 3
Data Analysis Pandas, NumPy
Visualisation Matplotlib, Seaborn
Machine Learning Scikit-learn
Environment VS Code

🔍 Project Workflow
1. Data Cleaning & Preprocessing

Converted TotalCharges from object to numeric (11 rows had blank values — dropped)
No other null values found after cleaning
Encoded all categorical columns using LabelEncoder

2. Exploratory Data Analysis (EDA)
Key findings from the analysis:
Insight             Finding
Overall churn rate ~26.5% of customers churned
Contract type      Month-to-month customers churn at 3x the rate of annual contract holders
Tenure             Customers with tenure < 12 months churn at nearly double the rate of long-term customers
Monthly charges    High-charge customers (above mean ₹64.8) show significantly higher churn
Internet service   Fiber optic users churn more than DSL users

3. Machine Learning Models
Random Forest Classifier
Accuracy: 79.8%

              precision  recall  f1-score
Not Churned      0.85     0.90      0.87
Churned          0.65     0.52      0.58
Logistic Regression (class_weight=balanced)
Accuracy: 75.4%

              precision  recall  f1-score
Not Churned      0.88     0.76      0.82
Churned          0.55     0.73      0.63

🏆 Model Comparison

Random Forest wins on overall accuracy (79.8% vs 75.4%)
Logistic Regression wins on churn recall (0.73 vs 0.52) — better at catching actual churners
For a business use case, Logistic Regression is preferable because missing a churner is more costly than a false alarm

4. Feature Importance (Random Forest)
Top 5 features driving churn prediction:

TotalCharges — 20.6%
MonthlyCharges — 19.3%
tenure — 17.0%
Contract — 13.2%
PaymentMethod — 6.6%

📊 Visualisations
The following charts were generated during EDA:

Churn distribution (count plot)
Contract type vs Churn
Monthly charges distribution by churn status
Tenure distribution by churn status
Internet service vs Churn

💡 Key Business Insights

Target month-to-month customers with loyalty offers before month 12 — this is the highest risk window
High monthly charge customers need proactive outreach — consider bundle discounts
Fiber optic users churn more — investigate service quality issues in this segment
Longer tenure = lower churn — reward loyalty milestones to encourage retention

📁 Project Structure
customer-churn-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── churn_analysis.py
└── README.md


