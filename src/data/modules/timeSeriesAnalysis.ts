import type { MLModule } from '../../types';

export const timeSeriesAnalysis: MLModule = {
  id: 'time-series-analysis',
  title: 'Time Series Analysis',
  category: 'Supervised Learning',
  description: 'Model sequential temporal dependencies using lag features, rolling windows, and time-based splits.',
  formula: 'y_t = f(y_{t-1}, y_{t-2}, \\dots, t)',
  theory: `### Time Series Analysis & Feature Engineering

#### What is it?
A time series is a sequence of data points indexed in chronological order, typically taken at successive equally spaced points in time. Examples include stock prices, weather measurements, retail sales, or website traffic.
Time Series analysis involves understanding the underlying patterns of temporal data (trends, seasonality, cycles), while Time Series forecasting uses historical observations to predict future values. In machine learning, time series forecasting is often converted into a supervised regression problem by generating lag and rolling features.

#### Why do we need it?
1. **Dynamic Decision Making:** Businesses need forecasts to optimize inventory, staff, and budget.
2. **Temporal Autocorrelation:** Time series data violates the standard machine learning assumption of independent and identically distributed (i.i.d.) observations. Points close in time are highly correlated.
3. **Preventing Leakage:** Standard random cross-validation cannot be used on temporal data. Training must strictly precede validation to avoid predicting the past using the future (data leakage).

#### How does it work?
In competitive machine learning (Kaggle), forecasting is often framed as a tabular regression problem:
1. **Lag Features:** Creating columns containing values from previous time steps (e.g., $y_{t-1}, y_{t-7}$). This captures autocorrelation directly.
2. **Rolling Statistics:** Computing summary statistics (mean, std, min, max) over sliding windows (e.g., a 7-day rolling average of sales).
3. **Date-Part Features:** Extracting calendar values like day-of-week, month, year, or is_weekend to capture cyclical seasonality patterns.
4. **Validation (TimeSeriesSplit):** Splitting data chronologically. The test set is always in the future relative to the training set.

#### The Math Behind It

**Autoregressive (AR) Model:**
An autoregressive model of order $p$ predicts $y_t$ as a linear combination of its past $p$ values:
$$y_t = c + \\sum_{i=1}^p \\phi_i y_{t-i} + \\epsilon_t$$
where $\\epsilon_t$ is white noise.

**Rolling Average:**
A simple rolling average of window size $w$:
$$\text{MA}_t(w) = \frac{1}{w} \\sum_{i=0}^{w-1} y_{t-i}$$

**Stationarity:**
For time-series models to generalize, the data must be stationary—meaning its statistical properties (mean, variance, autocorrelation) do not change over time. Non-stationary data is transformed using **differencing**:
$$\\Delta y_t = y_t - y_{t-1}$$

#### Worked Example: Lag Features
Suppose we have daily sales: $[10, 12, 15, 11]$.
If we create a 1-day lag feature:
- Sales: $[10, 12, 15, 11]$
- Lag 1: $[NaN, 10, 12, 15]$
Now, we can predict today's sales using yesterday's sales as a tabular regression feature.

#### Common Pitfalls
1. **Future Leakage:** Creating rolling features or lags using future target values, or using future data during training.
2. **Spurious Regression:** Regressing two non-stationary variables that both trend upwards (e.g., global temperatures and grocery prices), resulting in a high $R^2$ even though they are completely unrelated.
3. **Improper Splitting:** Using standard \`train_test_split\` which randomly shuffles data, leaking future points into the training fold.

#### Python Implementation

\`\`\`python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

np.random.seed(42)
data = np.cumsum(np.random.randn(100))
model = ARIMA(data, order=(1,1,1))
result = model.fit()
print(result.summary())
\`\`\`
`,
  interactiveSummary: 'This interactive simulator demonstrates how to extract time-series patterns. Add coordinates along a temporal stream and observe lag and rolling window features change dynamically. See how time-based cross-validation prevents data leakage in regression.',
  simulatorId: 'time-series',
  quiz: [
    {
      id: 'ts_q1',
      question: 'What is stationarity in a time series?',
      options: [
        'A state where the mean, variance, and autocorrelation remain constant over time.',
        'When the data has zero variance.',
        'When predictions do not change between epochs.',
        'A time series that only contains weekends.'
      ],
      correctAnswer: 'A state where the mean, variance, and autocorrelation remain constant over time.',
      explanation: 'A stationary time series has statistical properties that do not depend on the time at which the series is observed. It is crucial for traditional forecasting models.'
    },
    {
      id: 'ts_q2',
      question: 'How do you prevent data leakage during cross-validation of time series data?',
      options: [
        'By using chronological splits where the training set strictly precedes the test set (TimeSeriesSplit).',
        'By random shuffling with train_test_split.',
        'By scaling features before splitting.',
        'By omitting lag features.'
      ],
      correctAnswer: 'By using chronological splits where the training set strictly precedes the test set (TimeSeriesSplit).',
      explanation: 'Chronological splits respect the temporal order, ensuring that we never use future data to predict past values, which represents severe data leakage.'
    },
    {
      id: 'ts_q3',
      question: 'What is a lag feature?',
      options: [
        'A feature representing a value of the target at a previous time step (e.g. y_t-1).',
        'The delay in computation during model inference.',
        'A feature with missing values.',
        'The difference between training and test loss.'
      ],
      correctAnswer: 'A feature representing a value of the target at a previous time step (e.g. y_t-1).',
      explanation: 'Lags are historical observations shifted by one or more time steps. They are essential features for converting forecasting into supervised learning.'
    },
    {
      id: 'ts_q4',
      question: 'Which method is commonly used to remove trends and make a non-stationary time series stationary?',
      options: [
        'Differencing (y_t - y_t-1)',
        'One-Hot Encoding',
        'L2 Regularization',
        'MinMax Scaling'
      ],
      correctAnswer: 'Differencing (y_t - y_t-1)',
      explanation: 'Differencing computes the differences between consecutive observations, which removes trend and seasonality, transforming non-stationary series into stationary ones.'
    },
    {
      id: 'ts_q5',
      question: 'What is autocorrelation?',
      options: [
        'The correlation of a variable with itself at different time steps.',
        'The automatic scaling of correlation coefficients.',
        'The correlation between the model weights.',
        'The correlation between features and categorical labels.'
      ],
      correctAnswer: 'The correlation of a variable with itself at different time steps.',
      explanation: 'Autocorrelation measures the degree of similarity between a given time series and a lagged version of itself over successive time intervals.'
    }
  ],
  coding: {
    tutorial: {
      title: 'Vectorized Lag Generation',
      description: 'Implement a vectorized function in NumPy to generate a lag feature (shifting values by a step). Fill shifted empty slots with NaN.',
      pseudoCode: '1. Accept a numpy array X and step size.\n2. Create an empty array of shape X filled with NaN.\n3. Copy X[0:-step] into the empty array from index step onwards.\n4. Return lagged array.',
      starterCode: `import numpy as np

def generate_lag(X, step=1):
    # TODO: Shift X by 'step' positions and fill the beginning with np.nan
    lagged = np.full_like(X, np.nan, dtype=float)
    return lagged

sales = np.array([100.0, 120.0, 150.0, 130.0, 160.0])
print("Lag-1:", generate_lag(sales, step=1))`,
      expectedOutput: 'Lag-1: [ nan 100. 120. 150. 130.]',
      solution: `import numpy as np

def generate_lag(X, step=1):
    lagged = np.full_like(X, np.nan, dtype=float)
    if step < len(X):
        lagged[step:] = X[:-step]
    return lagged

sales = np.array([100.0, 120.0, 150.0, 130.0, 160.0])
print("Lag-1:", generate_lag(sales, step=1))`,
      hints: [
        'Use np.full_like with shape X and fill with np.nan.',
        'Slice from step onwards: lagged[step:] = X[:-step].'
      ],
      testKeywords: [
        'full_like',
        'np.nan',
        '[:-step]'
      ]
    },
    project: {
      title: 'Sales Forecasting with Lags & Ridge',
      description: 'Create lag features from historical sales and train a Ridge regressor to forecast the next day\'s sales.',
      pseudoCode: '1. Shift sales to create Lag-1 and Lag-2 features.\n2. Drop rows containing NaNs.\n3. Fit a Ridge regression model.\n4. Forecast next value and print R^2 score.',
      starterCode: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

# Daily Sales stream
sales = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 16.0, 18.0, 15.0])

# TODO: Create Lag-1: Shifted by 1 (e.g. [nan, 10, 12, ...])
# TODO: Create Lag-2: Shifted by 2 (e.g. [nan, nan, 10, ...])
# TODO: Combine features, drop rows with NaNs (first 2 rows)
# TODO: Fit Ridge(alpha=1.0) on X = [Lag1, Lag2] and y = sales
# TODO: Print R^2 score rounded to 2 decimal places

r2 = 0.0
print("R^2 Score:", round(r2, 2))`,
      expectedOutput: 'R^2 Score: 0.81',
      solution: `import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sales = np.array([10.0, 12.0, 15.0, 11.0, 14.0, 16.0, 18.0, 15.0])

# Create lags manually
lag1 = sales[:-2][1:]       # indices 1 to 5 (shifted by 1)
lag2 = sales[:-2][:-1]      # indices 0 to 4 (shifted by 2)
y = sales[2:]               # indices 2 to 7 (target)

X = np.column_stack((lag1, lag2))

model = Ridge(alpha=1.0)
model.fit(X, y)
preds = model.predict(X)
r2 = r2_score(y, preds)
print("R^2 Score:", round(r2, 2))`,
      hints: [
        'Shift index mappings: target starts at index 2, Lag-1 is index 1 to 6, Lag-2 is index 0 to 5.',
        'Use np.column_stack to build feature matrix X.',
        'Fit Ridge model on X and y, and evaluate using r2_score.'
      ],
      testKeywords: [
        'Ridge',
        'column_stack',
        'fit',
        'r2_score'
      ]
    },
    assignment: {
      title: 'TimeSeriesSplit Validation',
      description: 'Kaggle challenge: Evaluate a regression model chronologically using `TimeSeriesSplit` cross-validation to prevent data leakage.',
      pseudoCode: '1. Set up TimeSeriesSplit with n_splits=3.\n2. Iterate through training/test splits.\n3. Fit a Ridge model for each split.\n4. Compute average validation MSE.\n5. Print mean validation MSE.',
      starterCode: `import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1, 12.0])

# TODO: Initialize TimeSeriesSplit with n_splits=3
# TODO: Loop over split indices using split(X)
# TODO: Train Ridge(alpha=1.0) on X_train, y_train
# TODO: Predict on X_test, compute MSE
# TODO: Print the mean validation MSE rounded to 2 decimal places

mean_mse = 0.0
print("Mean MSE:", round(mean_mse, 2))`,
      expectedOutput: 'Mean MSE: 0.03',
      solution: `import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
y = np.array([2.1, 4.0, 6.2, 8.0, 10.1, 12.0])

tscv = TimeSeriesSplit(n_splits=3)
mses = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mses.append(mean_squared_error(y_test, preds))

mean_mse = np.mean(mses)
print("Mean MSE:", round(mean_mse, 2))`,
      hints: [
        'Use tscv.split(X) to get train and test indices.',
        'Index matrices with slices: X_train = X[train_index].',
        'Append mean_squared_error to a list and compute np.mean at the end.'
      ],
      testKeywords: [
        'TimeSeriesSplit',
        'split',
        'mean_squared_error',
        'np.mean'
      ]
    }
  },
  interviewQuestions: [
    {
      question: 'Why is standard K-Fold cross validation inappropriate for time series forecasting?',
      answer: 'Standard K-fold cross validation shuffles the dataset and randomly assigns points to train and validation folds. In time series, this shatters temporal order and leaks future values (e.g. y_t+1) into training to predict past values (e.g. y_t). Chronological splitting like TimeSeriesSplit is required to ensure validation strictly follows training in time.',
      companyTags: ['Kaggle', 'Uber'],
      difficulty: 'Medium'
    },
    {
      question: 'What is a stationary process, and why is it desirable for time series models?',
      answer: 'A stationary process is a time series whose statistical properties (mean, variance, covariance) are constant over time. It is desirable because most forecasting models assume that historical relationships will continue into the future. If statistical properties are changing (non-stationary), the model cannot generalize. Non-stationary data is transformed using differencing or log transforms.',
      companyTags: ['Faang', 'Jane Street'],
      difficulty: 'Hard'
    },
    {
      question: 'Explain how you would engineer features to capture weekly and yearly seasonality in a daily sales dataset.',
      answer: 'To capture weekly seasonality, I would extract date-part cyclical features: day of week (0-6), is_weekend (binary), or project them using sine/cosine transformations (e.g. sin(2*pi*day/7)). To capture yearly seasonality, I would extract month (1-12) or day-of-year (1-365) and apply cyclical sine/cosine encoding, or create 365-day lag features.',
      companyTags: ['Amazon', 'Walmart'],
      difficulty: 'Medium'
    }
  ]
};
