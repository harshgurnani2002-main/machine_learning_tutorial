export const kaggleStoreSalesForecastingProject = {
    id: 'kaggle-store-sales-forecasting-project',
    title: 'Kaggle Project: Store Sales Forecasting',
    category: 'Kaggle Real-World Projects',
    description: 'Forecast daily store sales using time-series features, holiday effects, and leakage-safe temporal validation.',
    formula: 'y_t = f(\\text{lag}_1, \\text{lag}_7, \\text{rolling\\_mean}_7, \\text{promo}_t, \\text{dow}_t)',
    theory: `### Real-World Problem
Forecast short-term demand for retail operations planning (inventory, staffing, logistics).

### Dataset
- Kaggle: Store Sales - Time Series Forecasting (or equivalent)
- Typical features: date, store_nbr, family, onpromotion, sales, holiday events

### Notebook Coding Approach
1. Sort data by date and entity keys.
2. Create lag and rolling features by store-family group.
3. Build calendar features: day-of-week, month, holiday flags.
4. Split chronologically (no random split).
5. Train baseline Ridge/LightGBM regressor.
6. Evaluate with MAE/RMSE on forward validation windows.
7. Retrain and generate test forecasts.

### Real-world extension
Forecast pipelines should be backtested across multiple historical windows, not a single split, to estimate deployment stability.`,
    interactiveSummary: 'A practical forecasting module emphasizing lag features and strict time-aware validation.',
    quiz: [
        {
            id: 'kssfp_q1',
            question: 'Why is random train_test_split unsafe for sales forecasting?',
            options: [
                'It leaks future information into training and inflates metrics.',
                'It makes models linear.',
                'It drops lag features.',
                'It cannot handle categorical data.'
            ],
            correctAnswer: 'It leaks future information into training and inflates metrics.',
            explanation: 'Temporal order must be respected to match real deployment conditions.'
        },
        {
            id: 'kssfp_q2',
            question: 'What does lag_7 capture in daily data?',
            options: [
                'Weekly recurrence signal from the same weekday last week.',
                'Model regularization strength.',
                'Target leakage from future rows.',
                'Noise-only pattern.'
            ],
            correctAnswer: 'Weekly recurrence signal from the same weekday last week.',
            explanation: 'Lag 7 often captures weekly seasonality in retail demand.'
        },
        {
            id: 'kssfp_q3',
            question: 'Which validation design is preferred?',
            options: ['Rolling/expanding time split', 'Bootstrap random split', 'Single shuffled holdout', 'KMeans folds'],
            correctAnswer: 'Rolling/expanding time split',
            explanation: 'Backtesting with rolling windows better estimates future forecasting behavior.'
        }
    ],
    notebookCells: [
        {
            id: 'kssfp_overview',
            type: 'markdown',
            title: 'Notebook overview',
            summary: 'Goal: forecast daily sales using lag features and strict time-based validation.'
        },
        {
            id: 'kssfp_imports',
            type: 'code',
            title: 'Imports',
            summary: 'Load libraries for time series feature engineering.',
            code: `import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error`
        },
        {
            id: 'kssfp_load',
            type: 'code',
            title: 'Load data',
            summary: 'Read train CSV and confirm shape.',
            code: `df = pd.read_csv('train.csv')
print(df.shape)`
        },
        {
            id: 'kssfp_dates',
            type: 'code',
            title: 'Parse dates and sort',
            summary: 'Ensure chronological order for feature generation.',
            code: `df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)`
        },
        {
            id: 'kssfp_calendar',
            type: 'code',
            title: 'Calendar features',
            summary: 'Add day-of-week and month features.',
            code: `df['dow'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month`
        },
        {
            id: 'kssfp_lags',
            type: 'code',
            title: 'Lag + rolling features',
            summary: 'Create lag_1 and rolling_mean_7 without leakage.',
            code: `df['lag_1'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1)
df['rolling_mean_7'] = df.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).mean().reset_index(level=[0, 1], drop=True)`,
            stageId: 'tutorial'
        },
        {
            id: 'kssfp_dropna',
            type: 'code',
            title: 'Drop initial NA rows',
            summary: 'Remove rows created by lag/rolling windows.',
            code: `df = df.dropna().reset_index(drop=True)`
        },
        {
            id: 'kssfp_split',
            type: 'code',
            title: 'Temporal validation split',
            summary: 'Split chronologically using a cutoff date.',
            code: `feature_cols = ['lag_1', 'rolling_mean_7', 'onpromotion', 'dow', 'month']
cutoff = '2017-07-01'

train_mask = df['date'] < cutoff
X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, 'sales']
X_val = df.loc[~train_mask, feature_cols]
y_val = df.loc[~train_mask, 'sales']`,
            stageId: 'project'
        },
        {
            id: 'kssfp_baseline',
            type: 'code',
            title: 'Baseline model',
            summary: 'Train a Ridge model and evaluate MAE.',
            code: `model = Ridge(alpha=1.0)
model.fit(X_train, y_train)
pred = model.predict(X_val)
mae = mean_absolute_error(y_val, pred)
print(round(mae, 3))`
        },
        {
            id: 'kssfp_backtest',
            type: 'code',
            title: 'Backtest rolling windows',
            summary: 'Evaluate MAE across multiple cutoffs.',
            code: `cutoffs = ['2017-05-01', '2017-06-01', '2017-07-01']
maes = []

for cutoff in cutoffs:
    train_mask = df['date'] < cutoff
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'sales']
    X_val = df.loc[~train_mask, feature_cols]
    y_val = df.loc[~train_mask, 'sales']
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    maes.append(mean_absolute_error(y_val, pred))

print(np.mean(maes))`,
            stageId: 'assignment'
        },
        {
            id: 'kssfp_submission',
            type: 'code',
            title: 'Submission export',
            summary: 'Train on full data and generate test forecasts.',
            code: `test = pd.read_csv('test.csv')

combined = pd.concat([df, test], ignore_index=True, sort=False)
combined['date'] = pd.to_datetime(combined['date'])
combined = combined.sort_values(['store_nbr', 'family', 'date']).reset_index(drop=True)

combined['dow'] = combined['date'].dt.dayofweek
combined['month'] = combined['date'].dt.month
combined['lag_1'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1)
combined['rolling_mean_7'] = combined.groupby(['store_nbr', 'family'])['sales'].shift(1).rolling(7).mean().reset_index(level=[0, 1], drop=True)

full_model = Ridge(alpha=1.0)
full_model.fit(df[feature_cols], df['sales'])

test_features = combined[combined['sales'].isna()].dropna(subset=['lag_1', 'rolling_mean_7'])
test_pred = full_model.predict(test_features[feature_cols])

submission = pd.DataFrame({'id': test_features['id'], 'sales': test_pred})
submission.to_csv('submission.csv', index=False)
print(submission.head())`
        }
    ],
    coding: {
        tutorial: {
            title: 'Create Lag and Rolling Mean Features',
            description: 'Generate lag_1 and rolling_mean_7 from a sales series.',
            pseudoCode: '1. Shift sales by 1 for lag_1.\n2. Shift then rolling(7).mean for rolling_mean_7.',
            starterCode: `import pandas as pd

df = pd.DataFrame({'sales': [10, 12, 13, 15, 11, 10, 14, 16]})
# TODO
df['lag_1'] = df['sales']
df['rolling_mean_7'] = df['sales']
print('lag_1' in df.columns and 'rolling_mean_7' in df.columns)`,
            expectedOutput: 'True',
            solution: `import pandas as pd

df = pd.DataFrame({'sales': [10, 12, 13, 15, 11, 10, 14, 16]})
df['lag_1'] = df['sales'].shift(1)
df['rolling_mean_7'] = df['sales'].shift(1).rolling(7).mean()
print('lag_1' in df.columns and 'rolling_mean_7' in df.columns)`,
            hints: ['Shift first, then rolling to avoid peeking current target.'],
            testKeywords: ['shift', 'rolling']
        },
        project: {
            title: 'Temporal Validation Split',
            description: 'Split features and targets chronologically using a cutoff date.',
            pseudoCode: '1. Build boolean mask date < cutoff.\n2. Create X_train/X_valid and y_train/y_valid.',
            starterCode: `# assume df has date, features, sales
cutoff = '2017-07-01'
# TODO
print(cutoff < '2099-01-01')`,
            expectedOutput: 'True',
            solution: `# assume df has date, features, sales
cutoff = '2017-07-01'
train_mask = df['date'] < cutoff
X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, 'sales']
X_valid = df.loc[~train_mask, feature_cols]
y_valid = df.loc[~train_mask, 'sales']
print(cutoff < '2099-01-01')`,
            hints: ['Never shuffle for forecasting split.'],
            testKeywords: ['loc', 'date', 'cutoff']
        },
        assignment: {
            title: 'Backtest Multiple Forecast Windows',
            description: 'Evaluate model MAE across multiple chronological cutoffs and report average MAE.',
            pseudoCode: '1. Loop over cutoff dates.\n2. Train model on past, validate on next window.\n3. Store MAE and average.',
            starterCode: `import numpy as np
maes = []
# TODO append mae values from each fold
print(len(maes) >= 0)`,
            expectedOutput: 'True',
            solution: `import numpy as np
maes = []
for cutoff in cutoffs:
    train_mask = df['date'] < cutoff
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, 'sales']
    X_valid = df.loc[~train_mask, feature_cols]
    y_valid = df.loc[~train_mask, 'sales']

    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    mae = np.mean(np.abs(y_valid - pred))
    maes.append(mae)

print(len(maes) >= 0)`,
            hints: ['Use consistent validation window size if possible.'],
            testKeywords: ['mean', 'abs', 'append']
        }
    },
    interviewQuestions: [
        {
            question: 'How would you detect leakage in a forecasting feature pipeline?',
            answer: 'I audit feature timestamps and ensure every feature for date t is computed from data <= t-1. I also compare offline metrics between strict and permissive pipelines; large jumps often indicate leakage.',
            companyTags: ['Amazon', 'Walmart'],
            difficulty: 'Advanced'
        }
    ]
};
