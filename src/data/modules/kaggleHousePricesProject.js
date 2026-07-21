export const kaggleHousePricesProject = {
    id: 'kaggle-house-prices-project',
    title: 'Kaggle Project: House Price Regression',
    category: 'Kaggle Real-World Projects',
    description: 'Predict house sale prices with robust feature handling, log-target modeling, and tree-based ensembles.',
    formula: '\\hat{y} = f(x), \\; \\text{optimize RMSE}(\\log(y), \\log(\\hat{y}))',
    theory: `### Real-World Problem
Predict residential sale prices from mixed numerical and categorical housing attributes.

### Dataset
- Kaggle: House Prices - Advanced Regression Techniques
- Files: train.csv, test.csv

### Notebook Coding Approach
1. Inspect skewed target and use log1p transform.
2. Handle missing values by column type and domain defaults.
3. Engineer quality/age/interactions (e.g., TotalSF).
4. Encode categories via one-hot.
5. Train and compare models:
   - Ridge/Lasso baseline
   - GradientBoostingRegressor or XGBoost/LightGBM
6. Evaluate with cross-validated RMSE on log target.
7. Blend top models and export submission.

### Real-world connection
This mirrors pricing models in real-estate tech where robust preprocessing + careful validation beats model complexity alone.`,
    interactiveSummary: 'A practical end-to-end regression pipeline with target transformation and ensemble comparison.',
    quiz: [
        {
            id: 'khp_q1',
            question: 'Why is log-transforming sale price useful?',
            options: [
                'It stabilizes variance and reduces heavy right-tail skew.',
                'It converts regression into classification.',
                'It removes all outliers automatically.',
                'It guarantees better leaderboard rank.'
            ],
            correctAnswer: 'It stabilizes variance and reduces heavy right-tail skew.',
            explanation: 'House prices are often right-skewed; log transform makes optimization easier and metrics more stable.'
        },
        {
            id: 'khp_q2',
            question: 'Which validation metric does this Kaggle competition primarily use?',
            options: ['RMSE on log price', 'Accuracy', 'AUC', 'MAE on raw labels'],
            correctAnswer: 'RMSE on log price',
            explanation: 'The competition score is based on error after log transformation of sale price.'
        },
        {
            id: 'khp_q3',
            question: 'What is a safe first strategy for missing categorical values?',
            options: [
                'Impute with most frequent value or explicit "None" label.',
                'Drop all rows with any nulls.',
                'Fill all with zero.',
                'Use target mean directly from test data.'
            ],
            correctAnswer: 'Impute with most frequent value or explicit "None" label.',
            explanation: 'This preserves rows and avoids leakage while keeping semantics for missing categories.'
        }
    ],
    notebookCells: [
        {
            id: 'khp_overview',
            type: 'markdown',
            title: 'Notebook overview',
            summary: 'Goal: predict SalePrice with log-target RMSE and a leakage-safe preprocessing pipeline.'
        },
        {
            id: 'khp_imports',
            type: 'code',
            title: 'Imports',
            summary: 'Load libraries for preprocessing and modeling.',
            code: `import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor`
        },
        {
            id: 'khp_load_data',
            type: 'code',
            title: 'Load train/test data',
            summary: 'Read CSVs and confirm shapes.',
            code: `import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print(train.shape, test.shape)`
        },
        {
            id: 'khp_target_log',
            type: 'code',
            title: 'Log transform target',
            summary: 'Apply log1p to stabilize variance and keep inverse with expm1.',
            code: `import numpy as np

y = train['SalePrice']
y_log = np.log1p(y)
print(round(y_log.mean(), 4))`,
            stageId: 'tutorial'
        },
        {
            id: 'khp_missing',
            type: 'code',
            title: 'Missing value scan',
            summary: 'Rank columns by missing fraction to plan imputation.',
            code: `missing = train.isna().mean().sort_values(ascending=False)
print(missing.head(8))`
        },
        {
            id: 'khp_features',
            type: 'code',
            title: 'Feature engineering',
            summary: 'Create TotalSF and Age features for stronger signal.',
            code: `for df in [train, test]:
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Age'] = df['YrSold'] - df['YearBuilt']`
        },
        {
            id: 'khp_split',
            type: 'code',
            title: 'Train/validation split',
            summary: 'Split features and log target for quick validation.',
            code: `X = train.drop(columns=['SalePrice'])
X_train, X_val, y_train, y_val = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)`
        },
        {
            id: 'khp_preprocess',
            type: 'code',
            title: 'Preprocessing pipeline',
            summary: 'Impute numeric features and one-hot encode categoricals.',
            code: `num_cols = X.select_dtypes(include=['number']).columns
cat_cols = X.select_dtypes(exclude=['number']).columns

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])`
        },
        {
            id: 'khp_models',
            type: 'code',
            title: 'Model comparison with CV',
            summary: 'Compare Ridge vs Gradient Boosting using CV RMSE.',
            code: `def cv_rmse(model):
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    scores = cross_val_score(pipe, X, y_log, cv=5, scoring='neg_root_mean_squared_error')
    return float(-scores.mean())

ridge_rmse = cv_rmse(Ridge(alpha=10.0))
gbr_rmse = cv_rmse(GradientBoostingRegressor(random_state=42))

print('Ridge RMSE:', round(ridge_rmse, 4))
print('GBR RMSE:', round(gbr_rmse, 4))`,
            stageId: 'project'
        },
        {
            id: 'khp_submission',
            type: 'code',
            title: 'Blend models and submit',
            summary: 'Fit both models, blend predictions in log-space, and export submission.',
            code: `ridge = Ridge(alpha=10.0)
gbr = GradientBoostingRegressor(random_state=42)

ridge_pipe = Pipeline([('prep', preprocessor), ('model', ridge)])
gbr_pipe = Pipeline([('prep', preprocessor), ('model', gbr)])

ridge_pipe.fit(X, y_log)
gbr_pipe.fit(X, y_log)

pred_ridge_log = ridge_pipe.predict(test)
pred_gbr_log = gbr_pipe.predict(test)

blend_log = 0.5 * pred_ridge_log + 0.5 * pred_gbr_log
final_price = np.expm1(blend_log)

submission = pd.DataFrame({
    'Id': test['Id'],
    'SalePrice': final_price
})
submission.to_csv('submission.csv', index=False)
print(submission.head())`,
            stageId: 'assignment'
        }
    ],
    coding: {
        tutorial: {
            title: 'Log Target Transformation',
            description: 'Apply log1p transformation to regression targets and invert using expm1.',
            pseudoCode: '1. Use np.log1p on y.\n2. Recover with np.expm1.\n3. Verify approximate equality.',
            starterCode: `import numpy as np

y = np.array([100000.0, 150000.0, 250000.0])
# TODO: log transform and invert
y_log = y
y_back = y
print(np.allclose(y, y_back))`,
            expectedOutput: 'True',
            solution: `import numpy as np

y = np.array([100000.0, 150000.0, 250000.0])
y_log = np.log1p(y)
y_back = np.expm1(y_log)
print(np.allclose(y, y_back))`,
            hints: ['Use numerically stable pair log1p/expm1.'],
            testKeywords: ['log1p', 'expm1']
        },
        project: {
            title: 'Cross-Validated RMSE with Pipeline',
            description: 'Compute CV RMSE for a regularized regression model after preprocessing.',
            pseudoCode: '1. Build preprocessing + Ridge pipeline.\n2. Use cross_val_score with neg RMSE.\n3. Convert sign and average.',
            starterCode: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# assume preprocessor, X_train, y_log exist
pipe = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=10.0))])
# TODO: fill scores and rmse
rmse = 0.0
print(round(rmse, 4))`,
            expectedOutput: '0.1350',
            solution: `import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# assume preprocessor, X_train, y_log exist
pipe = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=10.0))])
scores = cross_val_score(pipe, X_train, y_log, cv=5, scoring='neg_root_mean_squared_error')
rmse = float(-scores.mean())
print(round(rmse, 4))`,
            hints: ['For losses, sklearn often returns negative values by convention.'],
            testKeywords: ['neg_root_mean_squared_error', 'cross_val_score']
        },
        assignment: {
            title: 'Blend Two Regressors for Submission',
            description: 'Average predictions from two trained regressors and convert back from log-space.',
            pseudoCode: '1. Predict log prices from model A/B.\n2. Average predictions.\n3. expm1 to original price scale.',
            starterCode: `import numpy as np

# assume pred_a_log and pred_b_log are numpy arrays
pred_a_log = np.array([11.5, 12.0])
pred_b_log = np.array([11.6, 11.9])
# TODO
final_price = np.array([])
print(final_price.shape[0])`,
            expectedOutput: '2',
            solution: `import numpy as np

pred_a_log = np.array([11.5, 12.0])
pred_b_log = np.array([11.6, 11.9])
blend_log = 0.5 * pred_a_log + 0.5 * pred_b_log
final_price = np.expm1(blend_log)
print(final_price.shape[0])`,
            hints: ['Blend in log-space before inverse transform.'],
            testKeywords: ['np.expm1']
        }
    },
    interviewQuestions: [
        {
            question: 'Why do simple linear baselines still matter in advanced Kaggle regression?',
            answer: 'They are leakage checks, stability anchors, and useful ensembling members. A weak but stable baseline often reveals pipeline errors early.',
            companyTags: ['Zillow', 'Uber'],
            difficulty: 'Intermediate'
        }
    ]
};
