export const kaggleCustomerChurnProject = {
    id: 'kaggle-customer-churn-project',
    title: 'Kaggle Project: Customer Churn Prediction',
    category: 'Kaggle Real-World Projects',
    description: 'Predict customer churn and translate model outputs into retention actions using explainable features.',
    formula: '\\text{lift} = \\frac{P(\\text{churn} \\mid \\text{top risk decile})}{P(\\text{churn overall})}',
    theory: `### Real-World Problem
Predict which customers are likely to leave so retention teams can intervene proactively.

### Dataset
- Kaggle: Telco Customer Churn (or equivalent churn datasets)
- Typical columns: tenure, monthly charges, contract type, support usage, payment method

### Notebook Coding Approach
1. Frame churn as binary classification.
2. Clean and encode categorical features.
3. Handle class imbalance and perform stratified split.
4. Train baseline Logistic Regression and Gradient Boosting.
5. Evaluate with AUC, recall, and lift at top-k.
6. Generate ranked churn-risk list with probabilities.
7. Add explainability with feature importance / SHAP summary.

### Real-world extension
Churn use cases require actionability: outputs should include risk score, reason signals, and suggested intervention segment.`,
    interactiveSummary: 'A business-facing churn module connecting model training to retention strategy.',
    quiz: [
        {
            id: 'kccp_q1',
            question: 'Why is top-k lift useful in churn projects?',
            options: [
                'It measures how concentrated churn is in the highest-risk segment.',
                'It replaces train-test split.',
                'It always equals AUC.',
                'It removes need for probabilities.'
            ],
            correctAnswer: 'It measures how concentrated churn is in the highest-risk segment.',
            explanation: 'Retention campaigns usually target limited customers, so lift in top deciles is operationally meaningful.'
        },
        {
            id: 'kccp_q2',
            question: 'Which output is most useful for retention teams?',
            options: [
                'Ranked list with churn probability and key drivers.',
                'Only model coefficients table.',
                'Only confusion matrix.',
                'Only training accuracy.'
            ],
            correctAnswer: 'Ranked list with churn probability and key drivers.',
            explanation: 'Teams need actionable targeting, not just aggregate metrics.'
        },
        {
            id: 'kccp_q3',
            question: 'Why keep a logistic baseline when using boosted trees?',
            options: [
                'To maintain interpretability and sanity-check performance.',
                'Because tree models cannot predict probabilities.',
                'Because logistic regression is always best.',
                'To avoid validation completely.'
            ],
            correctAnswer: 'To maintain interpretability and sanity-check performance.',
            explanation: 'Baselines help detect bugs and support interpretable comparisons.'
        }
    ],
    notebookCells: [
        {
            id: 'kccp_overview',
            type: 'markdown',
            title: 'Notebook overview',
            summary: 'Goal: predict churn risk, rank customers, and export a retention list.'
        },
        {
            id: 'kccp_imports',
            type: 'code',
            title: 'Imports',
            summary: 'Load libraries for preprocessing and modeling.',
            code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score`
        },
        {
            id: 'kccp_load',
            type: 'code',
            title: 'Load data',
            summary: 'Read churn dataset and confirm shape.',
            code: `df = pd.read_csv('churn.csv')
print(df.shape)`
        },
        {
            id: 'kccp_clean',
            type: 'code',
            title: 'Basic cleaning',
            summary: 'Convert TotalCharges to numeric and handle missing rows.',
            code: `df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])`
        },
        {
            id: 'kccp_encode',
            type: 'code',
            title: 'Encode categoricals',
            summary: 'One-hot encode categorical columns.',
            code: `cat_cols = df.select_dtypes(include=['object']).columns
cat_cols = [c for c in cat_cols if c not in ['customerID', 'Churn']]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)`
        },
        {
            id: 'kccp_split',
            type: 'code',
            title: 'Train/validation split',
            summary: 'Split with stratification for robust evaluation.',
            code: `ids = df['customerID']
y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['customerID', 'Churn'])

X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
    X, y, ids, test_size=0.2, random_state=42, stratify=y
)`
        },
        {
            id: 'kccp_baseline',
            type: 'code',
            title: 'Baseline model comparison',
            summary: 'Train Logistic Regression and Gradient Boosting, then compare AUC.',
            code: `logit = LogisticRegression(max_iter=1000)
gb = GradientBoostingClassifier(random_state=42)

logit.fit(X_train, y_train)
gb.fit(X_train, y_train)

logit_proba = logit.predict_proba(X_val)[:, 1]
gb_proba = gb.predict_proba(X_val)[:, 1]

print('Logit AUC:', round(roc_auc_score(y_val, logit_proba), 3))
print('GB AUC:', round(roc_auc_score(y_val, gb_proba), 3))`
        },
        {
            id: 'kccp_rank',
            type: 'code',
            title: 'Rank churn risk',
            summary: 'Create a ranked list of validation customers by churn probability.',
            code: `ranked = pd.DataFrame({
    'customer_id': ids_val,
    'churn_proba': gb_proba
}).sort_values('churn_proba', ascending=False)

print(ranked.head())`,
            stageId: 'tutorial'
        },
        {
            id: 'kccp_lift',
            type: 'code',
            title: 'Lift at top 20%',
            summary: 'Compute churn lift for the highest-risk 20% segment.',
            code: `df_lift = pd.DataFrame({'y': y_val, 'p': gb_proba}).sort_values('p', ascending=False)
k = max(1, int(0.2 * len(df_lift)))
top_rate = df_lift.head(k)['y'].mean()
base_rate = df_lift['y'].mean()
lift = float(top_rate / base_rate)
print(round(lift, 2))`,
            stageId: 'project'
        },
        {
            id: 'kccp_export',
            type: 'code',
            title: 'Retention export',
            summary: 'Assign segments and export a CRM-ready CSV.',
            code: `ranked['segment'] = pd.cut(
    ranked['churn_proba'],
    bins=[-1, 0.4, 0.7, 1.0],
    labels=['Low', 'Medium', 'High']
)
ranked.to_csv('retention_targets.csv', index=False)
print(ranked.head())`,
            stageId: 'assignment'
        }
    ],
    coding: {
        tutorial: {
            title: 'Churn Probability Ranking',
            description: 'Sort customers by predicted churn risk and return top N accounts.',
            pseudoCode: '1. Build DataFrame with customer id + proba.\n2. Sort descending by proba.\n3. Select head(N).',
            starterCode: `import pandas as pd

ids = [101, 102, 103]
proba = [0.2, 0.8, 0.6]
# TODO
out = pd.DataFrame()
print(out.shape[0])`,
            expectedOutput: '3',
            solution: `import pandas as pd

ids = [101, 102, 103]
proba = [0.2, 0.8, 0.6]
out = pd.DataFrame({'customer_id': ids, 'churn_proba': proba}).sort_values('churn_proba', ascending=False)
print(out.shape[0])`,
            hints: ['Use sort_values with ascending=False.'],
            testKeywords: ['DataFrame', 'sort_values']
        },
        project: {
            title: 'Compute Lift at Top 20%',
            description: 'Measure campaign effectiveness by computing lift in highest-risk segment.',
            pseudoCode: '1. Rank by proba descending.\n2. Mark top 20% rows.\n3. Compute churn rate in top segment and overall.',
            starterCode: `import numpy as np
import pandas as pd

# assume y_true (0/1) and y_proba arrays exist
lift = 0.0
print(lift >= 0.0)`,
            expectedOutput: 'True',
            solution: `import numpy as np
import pandas as pd

# assume y_true (0/1) and y_proba arrays exist
df = pd.DataFrame({'y': y_true, 'p': y_proba}).sort_values('p', ascending=False)
k = max(1, int(0.2 * len(df)))
top_rate = df.head(k)['y'].mean()
base_rate = df['y'].mean()
lift = float(top_rate / base_rate) if base_rate > 0 else 0.0
print(lift >= 0.0)`,
            hints: ['Use head(k) after sorting by score.'],
            testKeywords: ['sort_values', 'head', 'mean']
        },
        assignment: {
            title: 'Deploy-Ready Retention Export',
            description: 'Create an export containing top-risk customers with segment labels for CRM campaigns.',
            pseudoCode: '1. Rank by churn_proba.\n2. Create segment labels (High/Medium/Low).\n3. Save CSV for ops team.',
            starterCode: `import pandas as pd

scored = pd.DataFrame({'customer_id': [1,2], 'churn_proba': [0.9, 0.3]})
# TODO
scored['segment'] = ''
print('segment' in scored.columns)`,
            expectedOutput: 'True',
            solution: `import pandas as pd

scored = pd.DataFrame({'customer_id': [1, 2], 'churn_proba': [0.9, 0.3]})
scored = scored.sort_values('churn_proba', ascending=False)
scored['segment'] = pd.cut(
    scored['churn_proba'],
    bins=[-1, 0.4, 0.7, 1.0],
    labels=['Low', 'Medium', 'High']
)
scored.to_csv('retention_targets.csv', index=False)
print('segment' in scored.columns)`,
            hints: ['Use pd.cut for quick score banding.'],
            testKeywords: ['pd.cut', 'to_csv']
        }
    },
    interviewQuestions: [
        {
            question: 'What is the most important mistake to avoid in churn modeling?',
            answer: 'Label leakage from post-churn behavior. All features must be from the decision time window before churn occurs.',
            companyTags: ['Netflix', 'Spotify'],
            difficulty: 'Advanced'
        }
    ]
};
