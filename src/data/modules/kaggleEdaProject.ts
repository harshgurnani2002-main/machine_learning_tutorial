import type { MLModule } from '../../types';

export const kaggleEdaProject: MLModule = {
  id: 'kaggle-eda-project',
  title: 'Kaggle EDA & Data Preprocessing',
  category: 'Kaggle Real-World Projects',
  description: 'Master Exploratory Data Analysis — the single most critical step in any Kaggle competition. Learn systematic pipelines for missing values, distributions, correlations, and feature insights.',
  formula: '\text{Corr}(X,Y) = \frac{\text{Cov}(X,Y)}{\\sigma_X \\cdot \\sigma_Y}',
  interactiveSummary: 'Work through an interactive Kaggle-style notebook to perform EDA on a dataset. Execute each cell to reveal insights — distributions, correlations, missing values, and feature engineering opportunities.',
  simulatorId: 'kaggle-notebook',
  theory: `### Exploratory Data Analysis (EDA) — The Most Critical ML Step

#### What is EDA and Why Does It Matter?
Exploratory Data Analysis (EDA) is the systematic process of profiling a dataset before modeling. The goal is to understand what you have, identify problems, and uncover patterns that guide feature engineering.

In Kaggle competitions, the difference between a top-10 solution and an average one often comes down to EDA quality — not model complexity. Understanding your data deeply leads to better feature engineering, which often matters more than model choice.

#### The EDA Workflow — Step by Step

**Step 1: Initial Data Audit**
The very first thing you do with any new dataset:
\`\`\`python
df.shape          # (rows, columns) — how much data do we have?
df.dtypes         # Data types — numeric, categorical, datetime?
df.head(10)       # Eyeball the first 10 rows
df.tail(5)        # Check the end for any weirdness
df.describe()     # Count, mean, std, min, quartiles, max for numerics
df.info()         # Non-null counts and dtypes together
\`\`\`

**Step 2: Missing Value Analysis**
Missing values are one of the most common data quality issues:

\`\`\`python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
missing_df = missing_df[missing_df['count'] > 0].sort_values('percent', ascending=False)
print(missing_df)
\`\`\`

Understanding **why** data is missing is crucial:
- **MCAR (Missing Completely At Random)**: Missingness is unrelated to any variable. Safe to impute.
- **MAR (Missing At Random)**: Missingness depends on other observed variables. Conditional imputation.
- **MNAR (Missing Not At Random)**: Missingness depends on the missing value itself (e.g., rich people don't report income). The missing indicator IS informative — create a binary "was_missing" feature.

Imputation strategies:
- **Numeric**: Mean (normal distribution), Median (skewed), Mode, KNN imputation
- **Categorical**: Mode, "Unknown" category, model-based imputation

**Step 3: Univariate Analysis (One Variable at a Time)**
- **Numeric variables**: Plot histograms and box plots to check distribution shape, central tendency, and outliers
- **Categorical variables**: Use value_counts() and bar plots to see class frequencies
- **Target variable**: Always analyze the target distribution first! Is it balanced? Skewed? Multimodal?

Detecting skewness — if skewness > 0.5, consider a log transformation:
$$\text{Skewness} = \frac{E[(X-\\mu)^3]}{\\sigma^3}$$

For highly skewed positive features: apply $\\log(1 + x)$ transformation to normalize distribution.

**Step 4: Bivariate Analysis (Feature vs Target)**
Understand which features are predictive of the target:

\`\`\`python
# For numeric features vs numeric target:
df.corr()['target'].sort_values(ascending=False)

# For categorical features vs binary target:
df.groupby('category')['target'].mean()  # Shows survival rate by category

# For numeric features vs binary target:
df.groupby('target')['feature'].describe()  # Different distributions?
\`\`\`

**Pearson Correlation** measures linear relationship strength:
$$r = \frac{\\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\\sqrt{\\sum_i (x_i - \bar{x})^2 \\sum_i (y_i - \bar{y})^2}}$$

Values: -1 (perfect negative), 0 (no linear relationship), +1 (perfect positive).

**Step 5: Multivariate Analysis**
Examine interactions between multiple features:
- **Correlation heatmap**: Spot multicollinearity (two features highly correlated with each other)
- **Pair plots**: Visual matrix of all feature-vs-feature scatter plots
- **Pivot tables**: Cross-tabulations for categorical features

High inter-feature correlation (|r| > 0.9) suggests multicollinearity — consider dropping one feature.

**Step 6: Outlier Detection**
Two standard methods:
- **Z-Score**: Flag points where $|z| > 3$
- **IQR Method**: Flag points outside $[Q1 - 1.5 \times IQR, Q3 + 1.5 \times IQR]$

\`\`\`python
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5*IQR) | (df['feature'] > Q3 + 1.5*IQR)]
\`\`\`

**Step 7: Feature Engineering Opportunities**
EDA reveals features to create:
- **Interaction features**: Multiply or divide two correlated features (e.g., BMI = weight/height²)
- **Polynomial features**: Add $x^2$, $x^3$ to capture non-linear relationships
- **Datetime features**: Extract year, month, day, hour, day-of-week from timestamps
- **Aggregation features**: Count, mean, std of a feature grouped by a category
- **Binning**: Discretize continuous variables into meaningful buckets (age groups)
- **Text features**: Word count, character count, presence of keywords

**Step 8: The Seaborn Visualization Gallery**
Essential plots for EDA:
\`\`\`python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of a numeric feature
sns.histplot(df['age'], kde=True)

# Boxplot — shows median, quartiles, and outliers
sns.boxplot(x='survived', y='age', data=df)

# Violin plot — distribution shape per class
sns.violinplot(x='survived', y='fare', data=df)

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')

# Pairplot — all feature combinations
sns.pairplot(df, hue='survived')

# Count plot for categoricals
sns.countplot(x='pclass', hue='survived', data=df)
\`\`\`

#### Categorical Variable Encoding
Before modeling, categorical variables must be encoded:
- **Label Encoding**: Map categories to integers {A→0, B→1, C→2}. Use only for ordinal categories (small, medium, large) or tree-based models (XGBoost, Random Forest) which can handle ordinal numbers.
- **One-Hot Encoding**: Create a binary column for each category. Use for nominal categories in linear models/neural nets. Beware of the "dummy variable trap" (drop one column).
- **Target Encoding**: Replace each category with the mean target value of that category. Powerful but risks leakage — use cross-validation target encoding.
- **Frequency Encoding**: Replace category with its frequency in the dataset. Simple and effective.

#### Variance Inflation Factor (VIF) for Multicollinearity
VIF measures how much a feature's variance is inflated by its correlation with other features:

$$VIF_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the R-squared from regressing feature $j$ on all other features.
- VIF = 1: No collinearity
- VIF = 5-10: Moderate collinearity — investigate
- VIF > 10: Severe collinearity — consider dropping one of the correlated features
`,
  quiz: [
    {
      id: 'eda_q1',
      question: 'What does a correlation coefficient of r = 0.05 between a feature and the target variable tell you?',
      options: [
        'The feature is inversely correlated with the target',
        'There is almost no linear relationship between this feature and the target',
        'The feature perfectly predicts the target',
        'The feature should definitely be dropped'
      ],
      correctAnswer: 'There is almost no linear relationship between this feature and the target',
      explanation: 'r = 0.05 is near zero, indicating almost no linear relationship. However, this does NOT mean the feature is useless — it could have a non-linear relationship with the target. Always check scatter plots and try tree-based models which capture non-linear patterns.'
    },
    {
      id: 'eda_q2',
      question: 'What is the difference between MCAR and MNAR missing data? Why does it matter for imputation?',
      options: [
        'MCAR is easier to detect; MNAR requires domain knowledge — they have the same imputation strategy',
        'MCAR can be safely imputed; MNAR means missingness is informative and a binary "was_missing" feature should be added',
        'MCAR means you should drop rows; MNAR means you should drop columns',
        'MCAR only occurs in numerical features; MNAR only in categorical features'
      ],
      correctAnswer: 'MCAR can be safely imputed; MNAR means missingness is informative and a binary "was_missing" feature should be added',
      explanation: 'MCAR (Missing Completely At Random) is unrelated to other variables — simple imputation works. MNAR (Missing Not At Random) means the missing value itself carries information (e.g., high-income people leave salary blank). Imputing without capturing this signal discards information. Always add a binary "was_missing" indicator feature for MNAR columns.'
    },
    {
      id: 'eda_q3',
      question: 'When performing EDA, you find that feature A and feature B have a Pearson correlation of 0.97. What should you do?',
      options: [
        'Keep both features — more features always help',
        'Investigate for multicollinearity; consider dropping one, creating a ratio feature, or using PCA',
        'Drop both features immediately',
        'Replace both with their average'
      ],
      correctAnswer: 'Investigate for multicollinearity; consider dropping one, creating a ratio feature, or using PCA',
      explanation: 'r=0.97 means near-perfect linear correlation — they carry almost the same information. This causes multicollinearity which inflates coefficient variance in linear models (making interpretability impossible) and wastes computational resources. For tree-based models it is less harmful but still redundant. Best strategies: drop one, use their ratio/difference as a single feature, or use PCA to combine them.'
    },
    {
      id: 'eda_q4',
      question: 'You observe that the "income" feature has a skewness of 3.2. What transformation should you apply and why?',
      options: [
        'Standardize (z-score) to center the distribution around zero',
        'Apply log(1+x) transformation to compress the right tail and normalize the distribution',
        'Apply x² to exaggerate differences in the upper range',
        'No transformation needed — skewness does not affect model performance'
      ],
      correctAnswer: 'Apply log(1+x) transformation to compress the right tail and normalize the distribution',
      explanation: 'Skewness of 3.2 indicates a heavy right tail (a few very high incomes). Linear models assume normally distributed features. log(1+x) compresses large values while preserving small differences near zero. It is used instead of log(x) to handle zeros safely. Tree-based models (XGBoost, RF) are invariant to monotonic transformations, but linear models, SVMs, and KNNs benefit significantly.'
    },
    {
      id: 'eda_q5',
      question: 'What is the "dummy variable trap" in one-hot encoding, and how do you avoid it?',
      options: [
        'Having too many one-hot columns causes memory overflow — solved by using label encoding instead',
        'Perfect multicollinearity when all one-hot columns sum to 1 — solved by dropping one column (drop_first=True)',
        'One-hot encoding introduces ordinal relationships — solved by using target encoding',
        'One-hot encoding breaks for unseen categories at inference — solved by using a fixed vocabulary'
      ],
      correctAnswer: 'Perfect multicollinearity when all one-hot columns sum to 1 — solved by dropping one column (drop_first=True)',
      explanation: 'For a 3-category variable, 3 one-hot columns always sum to 1. The third column is perfectly predictable from the other two, creating perfect multicollinearity. In linear models, this means the design matrix is rank-deficient and coefficients are not unique. Solution: use pd.get_dummies(drop_first=True) or OneHotEncoder(drop="first") to drop one redundant column per variable.'
    },
    {
      id: 'eda_q6',
      question: 'What does a VIF (Variance Inflation Factor) of 12 indicate for a feature?',
      options: [
        'The feature has high predictive power (high F-statistic)',
        'Severe multicollinearity — this feature is 86% explained by other features',
        'The feature has 12 unique categories',
        'The feature should be log-transformed'
      ],
      correctAnswer: 'Severe multicollinearity — this feature is 86% explained by other features',
      explanation: 'VIF = 1/(1-R²) = 12 → R² = 1 - 1/12 ≈ 0.917. This means 91.7% of this feature\'s variance is explained by the other features — severe multicollinearity. VIF > 10 is the common threshold for "problematic." In linear regression, this makes coefficients unstable and hard to interpret. Consider dropping the feature, combining with correlated features, or applying Ridge regression which handles multicollinearity.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Complete EDA Pipeline on Titanic Data',
      description: 'Perform a systematic EDA on a synthetic Titanic-like dataset. Analyze shape, missing values, distributions, correlations, and class distributions — the exact workflow used in Kaggle competitions.',
      pseudoCode: `1. Load Titanic-like dataset (synthetic)
2. Print shape, dtypes, and head
3. Analyze missing values (count + percentage)
4. Print describe() for numeric features
5. Print correlation of each feature with Survived
6. Analyze survival rate by passenger class`,
      starterCode: `import pandas as pd
import numpy as np

# Create synthetic Titanic-like dataset
np.random.seed(42)
n = 891
df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    'Age': np.where(np.random.rand(n) < 0.2, np.nan, np.random.normal(30, 13, n).clip(1, 80)),
    'SibSp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.25, 0.1, 0.03, 0.02]),
    'Parch': np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.2, 0.1, 0.05]),
    'Fare': np.random.exponential(32, n),
    'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Embarked': np.where(np.random.rand(n) < 0.02, np.nan, np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]))
})
df['Sex_enc'] = (df['Sex'] == 'female').astype(int)
survive_prob = (0.1 + 0.35*df.Sex_enc + 0.15*(df.Pclass==1) - 0.15*(df.Pclass==3)).clip(0.05, 0.95)
df['Survived'] = (np.random.rand(n) < survive_prob).astype(int)

print("=== 1. SHAPE ===")
# TODO: Print shape

print("\n=== 2. MISSING VALUES ===")
# TODO: Print count and percentage of missing values for columns with any missing

print("\n=== 3. DESCRIPTIVE STATISTICS ===")
# TODO: Print describe() for numeric columns (exclude PassengerId)

print("\n=== 4. CORRELATION WITH TARGET ===")
# TODO: Print Pearson correlation of all numeric features with 'Survived', sorted descending

print("\n=== 5. SURVIVAL RATE BY CLASS ===")
# TODO: Print mean 'Survived' grouped by 'Pclass'`,
      expectedOutput: `=== 1. SHAPE ===
(891, 10)`,
      solution: `import pandas as pd
import numpy as np

np.random.seed(42)
n = 891
df = pd.DataFrame({
    'PassengerId': range(1, n+1),
    'Pclass': np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55]),
    'Age': np.where(np.random.rand(n) < 0.2, np.nan, np.random.normal(30, 13, n).clip(1, 80)),
    'SibSp': np.random.choice([0, 1, 2, 3, 4], n, p=[0.6, 0.25, 0.1, 0.03, 0.02]),
    'Parch': np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.2, 0.1, 0.05]),
    'Fare': np.random.exponential(32, n),
    'Sex': np.random.choice(['male', 'female'], n, p=[0.65, 0.35]),
    'Embarked': np.where(np.random.rand(n) < 0.02, np.nan, np.random.choice(['S', 'C', 'Q'], n, p=[0.72, 0.19, 0.09]))
})
df['Sex_enc'] = (df['Sex'] == 'female').astype(int)
survive_prob = (0.1 + 0.35*df.Sex_enc + 0.15*(df.Pclass==1) - 0.15*(df.Pclass==3)).clip(0.05, 0.95)
df['Survived'] = (np.random.rand(n) < survive_prob).astype(int)

print("=== 1. SHAPE ===")
print(df.shape)

print("\n=== 2. MISSING VALUES ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_report = pd.DataFrame({'Count': missing, 'Percent': missing_pct})
print(missing_report[missing_report['Count'] > 0].sort_values('Percent', ascending=False))

print("\n=== 3. DESCRIPTIVE STATISTICS ===")
print(df.drop('PassengerId', axis=1).describe().round(2))

print("\n=== 4. CORRELATION WITH TARGET ===")
numeric_cols = df.select_dtypes(include='number').drop(columns=['PassengerId', 'Survived'])
print(numeric_cols.corrwith(df['Survived']).sort_values(ascending=False).round(4))

print("\n=== 5. SURVIVAL RATE BY CLASS ===")
print(df.groupby('Pclass')['Survived'].mean().round(3))`,
      hints: [
        'df.isnull().sum() counts nulls per column',
        'df.describe() auto-selects numeric columns',
        'numeric_cols.corrwith(df[\'Survived\']) computes correlation with target',
        'df.groupby(\'Pclass\')[\'Survived\'].mean() gives survival rate per class'
      ],
      testKeywords: ['isnull', 'describe', 'corrwith', 'groupby', 'sort_values']
    },
    project: {
      title: 'EDA on House Prices Dataset',
      description: 'Perform a full EDA on a synthetic House Prices dataset — the classic Kaggle regression challenge. Analyze price distribution, find top correlated features, detect outliers, and identify missing value patterns.',
      pseudoCode: `1. Create House Prices dataset (SalePrice, GrLivArea, TotalBsmtSF, GarageArea, etc.)
2. Check for skewness in SalePrice → apply log transform
3. Find top 10 features by correlation with SalePrice
4. Detect outliers using IQR method
5. Create new feature: TotalSF = GrLivArea + TotalBsmtSF
6. Check if TotalSF improves correlation with SalePrice`,
      starterCode: `import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

# Synthetic house prices dataset
df = pd.DataFrame({
    'SalePrice': np.random.lognormal(mean=12, sigma=0.4, size=n),
    'GrLivArea': np.random.normal(1500, 400, n).clip(500, 4000),
    'TotalBsmtSF': np.random.normal(1000, 300, n).clip(0, 2500),
    'GarageArea': np.random.normal(480, 150, n).clip(0, 1200),
    'YearBuilt': np.random.randint(1950, 2010, n),
    'OverallQual': np.random.choice(range(1, 11), n),
    'BedroomAbvGr': np.random.choice([1,2,3,4,5], n, p=[0.05, 0.2, 0.5, 0.2, 0.05]),
    'FullBath': np.random.choice([1,2,3], n, p=[0.3, 0.6, 0.1]),
    'LotArea': np.random.lognormal(mean=9, sigma=0.5, size=n),
    'Neighborhood': np.random.choice(['A','B','C','D','E'], n)
})
# Add some noise correlation
df['SalePrice'] += df['GrLivArea'] * 80 + df['OverallQual'] * 10000

# TODO: Print skewness of SalePrice
# TODO: Print skewness of log(SalePrice)
# TODO: Find and print top 10 correlations with SalePrice
# TODO: Detect and count outliers in SalePrice using IQR
# TODO: Create TotalSF = GrLivArea + TotalBsmtSF and print its correlation with SalePrice`,
      expectedOutput: `SalePrice skewness:`,
      solution: `import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'SalePrice': np.random.lognormal(mean=12, sigma=0.4, size=n),
    'GrLivArea': np.random.normal(1500, 400, n).clip(500, 4000),
    'TotalBsmtSF': np.random.normal(1000, 300, n).clip(0, 2500),
    'GarageArea': np.random.normal(480, 150, n).clip(0, 1200),
    'YearBuilt': np.random.randint(1950, 2010, n),
    'OverallQual': np.random.choice(range(1, 11), n),
    'BedroomAbvGr': np.random.choice([1,2,3,4,5], n, p=[0.05, 0.2, 0.5, 0.2, 0.05]),
    'FullBath': np.random.choice([1,2,3], n, p=[0.3, 0.6, 0.1]),
    'LotArea': np.random.lognormal(mean=9, sigma=0.5, size=n),
    'Neighborhood': np.random.choice(['A','B','C','D','E'], n)
})
df['SalePrice'] += df['GrLivArea'] * 80 + df['OverallQual'] * 10000

print("SalePrice skewness:", df['SalePrice'].skew().round(3))
print("log(SalePrice) skewness:", np.log1p(df['SalePrice']).skew().round(3))

numeric_df = df.select_dtypes(include='number')
corr = numeric_df.corrwith(df['SalePrice']).sort_values(ascending=False)
print("\nTop 10 correlations with SalePrice:")
print(corr.head(10).round(4))

Q1 = df['SalePrice'].quantile(0.25)
Q3 = df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['SalePrice'] < Q1 - 1.5*IQR) | (df['SalePrice'] > Q3 + 1.5*IQR)]
print(f"\nOutliers in SalePrice (IQR method): {len(outliers)} ({100*len(outliers)/n:.1f}%)")

df['TotalSF'] = df['GrLivArea'] + df['TotalBsmtSF']
total_sf_corr = df['TotalSF'].corr(df['SalePrice'])
orig_corr = df['GrLivArea'].corr(df['SalePrice'])
print(f"\nGrLivArea correlation with SalePrice: {orig_corr:.4f}")
print(f"TotalSF correlation with SalePrice: {total_sf_corr:.4f}")
print(f"Feature engineering improved correlation by: {total_sf_corr - orig_corr:.4f}")`,
      hints: [
        'df[\'column\'].skew() computes skewness',
        'np.log1p(x) = log(1+x), safe for zeros',
        'IQR = Q3 - Q1; outliers are < Q1-1.5*IQR or > Q3+1.5*IQR',
        'corrwith() computes correlation of all columns with a Series'
      ],
      testKeywords: ['skew', 'corrwith', 'quantile', 'IQR', 'corr', 'log1p']
    },
    assignment: {
      title: 'Feature Engineering Sprint',
      description: 'Given a raw dataset, apply a complete feature engineering pipeline: handle missing values, encode categoricals, create interaction features, remove highly correlated features, and train a baseline model before and after engineering to measure impact.',
      pseudoCode: `1. Load raw dataset with mixed types and missing values
2. Impute missing: median for numeric, mode for categorical
3. One-hot encode categorical columns
4. Create interaction features: TotalRooms, Age2023, FarePerPerson
5. Drop columns with >0.95 correlation with another column
6. Train LogisticRegression before and after engineering
7. Print accuracy improvement`,
      starterCode: `import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 500

# Raw messy dataset
df = pd.DataFrame({
    'age': np.where(np.random.rand(n) < 0.1, np.nan, np.random.normal(35, 12, n).clip(18, 80)),
    'fare': np.random.exponential(40, n),
    'fare_dup': None,  # Will duplicate fare (multicollinear)
    'pclass': np.random.choice([1,2,3], n),
    'sibsp': np.random.choice([0,1,2,3], n, p=[0.6,0.25,0.1,0.05]),
    'parch': np.random.choice([0,1,2], n, p=[0.7,0.2,0.1]),
    'sex': np.random.choice(['male','female'], n),
    'embarked': np.where(np.random.rand(n) < 0.05, np.nan, np.random.choice(['S','C','Q'], n))
})
df['fare_dup'] = df['fare'] + np.random.normal(0, 0.001, n)  # Near-duplicate
sex_enc = (df['sex'] == 'female').astype(int)
prob = (0.15 + 0.35*sex_enc - 0.12*(df['pclass']==3) + 0.01*df['parch']).clip(0.05, 0.95)
df['survived'] = (np.random.rand(n) < prob).astype(int)

y = df['survived']

# --- BASELINE (raw, no engineering) ---
# TODO: Encode sex, drop text/NaN columns, impute, scale, then cross_val_score

# --- AFTER FEATURE ENGINEERING ---
# TODO: 1. Impute age with median, embarked with mode
# TODO: 2. One-hot encode sex and embarked
# TODO: 3. Create FamilySize = sibsp + parch + 1
# TODO: 4. Create IsAlone = (FamilySize == 1).astype(int)
# TODO: 5. Drop fare_dup (high correlation with fare)
# TODO: 6. Scale and cross_val_score

# TODO: Print baseline vs engineered accuracy`,
      expectedOutput: `Baseline CV Accuracy:`,
      solution: `import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'age': np.where(np.random.rand(n) < 0.1, np.nan, np.random.normal(35, 12, n).clip(18, 80)),
    'fare': np.random.exponential(40, n),
    'fare_dup': None,
    'pclass': np.random.choice([1,2,3], n),
    'sibsp': np.random.choice([0,1,2,3], n, p=[0.6,0.25,0.1,0.05]),
    'parch': np.random.choice([0,1,2], n, p=[0.7,0.2,0.1]),
    'sex': np.random.choice(['male','female'], n),
    'embarked': np.where(np.random.rand(n) < 0.05, np.nan, np.random.choice(['S','C','Q'], n))
})
df['fare_dup'] = df['fare'] + np.random.normal(0, 0.001, n)
sex_enc = (df['sex'] == 'female').astype(int)
prob = (0.15 + 0.35*sex_enc - 0.12*(df['pclass']==3) + 0.01*df['parch']).clip(0.05, 0.95)
df['survived'] = (np.random.rand(n) < prob).astype(int)
y = df['survived']

# Baseline
baseline_df = df.copy()
baseline_df['sex_enc'] = (baseline_df['sex'] == 'female').astype(int)
baseline_df['age'].fillna(baseline_df['age'].median(), inplace=True)
X_base = baseline_df[['age', 'fare', 'pclass', 'sibsp', 'parch', 'sex_enc']].values
scaler = StandardScaler()
X_base = scaler.fit_transform(X_base)
base_score = cross_val_score(LogisticRegression(max_iter=500), X_base, y, cv=5).mean()
print(f"Baseline CV Accuracy: {base_score:.4f}")

# Feature Engineering
eng = df.copy()
eng['age'].fillna(eng['age'].median(), inplace=True)
eng['embarked'].fillna(eng['embarked'].mode()[0], inplace=True)
eng = pd.get_dummies(eng, columns=['sex', 'embarked'], drop_first=True)
eng['FamilySize'] = eng['sibsp'] + eng['parch'] + 1
eng['IsAlone'] = (eng['FamilySize'] == 1).astype(int)
eng = eng.drop(columns=['survived', 'fare_dup'])
X_eng = StandardScaler().fit_transform(eng.values)
eng_score = cross_val_score(LogisticRegression(max_iter=500), X_eng, y, cv=5).mean()
print(f"Engineered CV Accuracy: {eng_score:.4f}")
print(f"Improvement: +{eng_score - base_score:.4f}")`,
      hints: [
        'pd.get_dummies(df, columns=[\'sex\', \'embarked\'], drop_first=True) one-hot encodes',
        'fillna(df[col].median()) imputes numeric missing values',
        'FamilySize = SibSp + Parch + 1 (include the passenger themselves)',
        'cross_val_score returns an array — take .mean() for average accuracy'
      ],
      testKeywords: ['get_dummies', 'fillna', 'cross_val_score', 'FamilySize', 'StandardScaler', 'LogisticRegression']
    }
  },
  interviewQuestions: [
    {
      question: 'Walk me through your EDA process when you receive a new dataset for a Kaggle competition.',
      answer: 'I follow a structured 8-step process: (1) Initial audit — df.shape, dtypes, head(), info() to understand the dataset structure. (2) Missing value analysis — isnull().sum() with percentages, then categorize as MCAR/MAR/MNAR to choose imputation strategy. (3) Target analysis — always analyze the target first: is it skewed? Imbalanced? Bimodal? This determines evaluation metrics and modeling strategy. (4) Univariate analysis — histograms for numeric features, value_counts() for categoricals. Note skewness > 0.5 for log transformation. (5) Bivariate analysis — correlation with target, boxplots per class, group means. (6) Multivariate analysis — heatmap for inter-feature correlation, pairplot, VIF analysis. (7) Outlier detection — IQR or Z-score per feature. (8) Feature engineering hypotheses — based on domain knowledge and patterns found in EDA.',
      companyTags: ['Kaggle', 'Google', 'Amazon'],
      difficulty: 'Medium'
    },
    {
      question: 'How do you handle a categorical feature with 500 unique categories (high cardinality)?',
      answer: 'One-hot encoding would create 500 new columns — too many. Better strategies: (1) Frequency encoding: replace each category with its occurrence count or fraction in the training set. Simple and tree-friendly. (2) Target encoding: replace each category with the mean target value of that category. Powerful but requires careful cross-validation to avoid leakage (use KFold target encoding). (3) Binary encoding: convert category index to binary bits — much fewer columns than OHE. (4) Entity embeddings (if using neural networks): learn a dense vector representation per category. (5) Clustering: group rare categories into an "Other" bucket. Anything with frequency < 1% → "Rare". For Kaggle, target encoding + leave-one-out encoding is most commonly used for high-cardinality categoricals.',
      companyTags: ['Booking.com', 'Airbnb', 'Spotify'],
      difficulty: 'Hard'
    },
    {
      question: 'What is target leakage, and how does it manifest in EDA?',
      answer: 'Target leakage occurs when a feature contains information about the target that would not be available at prediction time. During EDA, a telltale sign is a feature with an unusually high correlation with the target (e.g., r=0.99). Examples: (1) "days_since_last_transaction" in a churn model — this is 0 for all churned customers, directly encoding the label. (2) "hospital_discharge_date" in a readmission model — only exists for patients who were admitted. (3) A timestamp feature that perfectly segments train/test splits by time. Prevention: always think about the prediction time — would this feature value be available when you make the prediction? If not, it is leakage. Use time-based cross-validation splits for time-series to prevent temporal leakage.',
      companyTags: ['Stripe', 'Square', 'Intuit'],
      difficulty: 'Hard'
    }
  ]
};
