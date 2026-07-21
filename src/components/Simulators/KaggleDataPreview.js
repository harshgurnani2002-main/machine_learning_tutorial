import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { Database, Table, Info, BarChart3 } from 'lucide-react';
const titanicData = {
    name: 'Titanic - Machine Learning from Disaster',
    source: 'Kaggle',
    rows: 891,
    columns: 12,
    target: 'Survived',
    challenge: 'Predict whether a passenger survived the Titanic disaster based on passenger attributes.',
    columnTypes: [
        { name: 'PassengerId', type: 'numeric' },
        { name: 'Survived', type: 'target' },
        { name: 'Pclass', type: 'numeric' },
        { name: 'Name', type: 'categorical' },
        { name: 'Sex', type: 'categorical' },
        { name: 'Age', type: 'numeric' },
        { name: 'SibSp', type: 'numeric' },
        { name: 'Parch', type: 'numeric' },
        { name: 'Fare', type: 'numeric' },
        { name: 'Embarked', type: 'categorical' },
    ],
    sampleRows: [
        { PassengerId: 1, Survived: 0, Pclass: 3, Name: 'Braund, Mr. Owen Harris', Sex: 'male', Age: 22, Fare: 7.25, Embarked: 'S' },
        { PassengerId: 2, Survived: 1, Pclass: 1, Name: 'Cumings, Mrs. John Bradley', Sex: 'female', Age: 38, Fare: 71.28, Embarked: 'C' },
        { PassengerId: 3, Survived: 1, Pclass: 3, Name: 'Heikkinen, Miss. Laina', Sex: 'female', Age: 26, Fare: 7.92, Embarked: 'S' },
        { PassengerId: 4, Survived: 1, Pclass: 1, Name: 'Futrelle, Mrs. Jacques Heath', Sex: 'female', Age: 35, Fare: 53.10, Embarked: 'S' },
        { PassengerId: 5, Survived: 0, Pclass: 3, Name: 'Allen, Mr. William Henry', Sex: 'male', Age: 35, Fare: 8.05, Embarked: 'S' },
        { PassengerId: 6, Survived: 0, Pclass: 3, Name: 'Moran, Mr. James', Sex: 'male', Age: NaN, Fare: 8.46, Embarked: 'Q' },
        { PassengerId: 7, Survived: 0, Pclass: 1, Name: 'McCarthy, Mr. Timothy J', Sex: 'male', Age: 54, Fare: 51.86, Embarked: 'S' },
        { PassengerId: 8, Survived: 0, Pclass: 3, Name: 'Palsson, Master. Gosta Leonard', Sex: 'male', Age: 2, Fare: 21.07, Embarked: 'S' },
    ],
};
const housePricesData = {
    name: 'House Prices - Advanced Regression Techniques',
    source: 'Kaggle',
    rows: 1460,
    columns: 81,
    target: 'SalePrice',
    challenge: 'Predict the sale price of residential homes in Ames, Iowa using 79 explanatory variables.',
    columnTypes: [
        { name: 'Id', type: 'numeric' },
        { name: 'MSSubClass', type: 'numeric' },
        { name: 'MSZoning', type: 'categorical' },
        { name: 'LotArea', type: 'numeric' },
        { name: 'Street', type: 'categorical' },
        { name: 'OverallQual', type: 'numeric' },
        { name: 'YearBuilt', type: 'numeric' },
        { name: 'SalePrice', type: 'target' },
    ],
    sampleRows: [
        { Id: 1, MSSubClass: 60, MSZoning: 'RL', LotArea: 8450, Street: 'Pave', OverallQual: 7, YearBuilt: 2003, SalePrice: 208500 },
        { Id: 2, MSSubClass: 20, MSZoning: 'RL', LotArea: 9600, Street: 'Pave', OverallQual: 6, YearBuilt: 1976, SalePrice: 181500 },
        { Id: 3, MSSubClass: 60, MSZoning: 'RL', LotArea: 11250, Street: 'Pave', OverallQual: 7, YearBuilt: 2001, SalePrice: 223500 },
        { Id: 4, MSSubClass: 70, MSZoning: 'RL', LotArea: 9550, Street: 'Pave', OverallQual: 7, YearBuilt: 1915, SalePrice: 140000 },
        { Id: 5, MSSubClass: 60, MSZoning: 'RL', LotArea: 14260, Street: 'Pave', OverallQual: 8, YearBuilt: 2000, SalePrice: 250000 },
        { Id: 6, MSSubClass: 50, MSZoning: 'RL', LotArea: 14115, Street: 'Pave', OverallQual: 5, YearBuilt: 1993, SalePrice: 143000 },
        { Id: 7, MSSubClass: 20, MSZoning: 'RL', LotArea: 10084, Street: 'Pave', OverallQual: 8, YearBuilt: 2004, SalePrice: 307000 },
        { Id: 8, MSSubClass: 60, MSZoning: 'RL', LotArea: 10382, Street: 'Pave', OverallQual: 6, YearBuilt: 1973, SalePrice: 200000 },
    ],
};
const fraudData = {
    name: 'Credit Card Fraud Detection',
    source: 'Kaggle',
    rows: 284807,
    columns: 31,
    target: 'Class',
    challenge: 'Identify fraudulent credit card transactions from a highly imbalanced dataset (0.17% fraud).',
    columnTypes: [
        { name: 'Time', type: 'numeric' },
        { name: 'V1', type: 'numeric' },
        { name: 'V2', type: 'numeric' },
        { name: 'V3', type: 'numeric' },
        { name: 'V4', type: 'numeric' },
        { name: 'V5', type: 'numeric' },
        { name: 'Amount', type: 'numeric' },
        { name: 'Class', type: 'target' },
    ],
    sampleRows: [
        { Time: 0, V1: -1.36, V2: -0.072, V3: 2.536, V4: 1.378, V5: -0.338, Amount: 149.62, Class: 0 },
        { Time: 0, V1: 1.192, V2: 0.266, V3: 0.166, V4: 0.448, V5: 0.060, Amount: 2.69, Class: 0 },
        { Time: 1, V1: -1.358, V2: -1.340, V3: 1.773, V4: 0.380, V5: -0.503, Amount: 378.66, Class: 0 },
        { Time: 1, V1: -0.966, V2: -0.185, V3: 1.793, V4: -0.863, V5: -0.010, Amount: 123.50, Class: 0 },
        { Time: 2, V1: -1.158, V2: 0.878, V3: 0.259, V4: 0.534, V5: -0.838, Amount: 69.99, Class: 0 },
        { Time: 2, V1: -0.426, V2: 0.961, V3: 1.141, V4: -0.223, V5: 0.051, Amount: 3.67, Class: 0 },
        { Time: 4, V1: 1.229, V2: 0.087, V3: -0.902, V4: 1.205, V5: 0.513, Amount: 93.92, Class: 0 },
        { Time: 7, V1: -0.570, V2: 2.201, V3: -1.112, V4: 1.463, V5: -1.103, Amount: 0.90, Class: 1 },
    ],
};
const churnData = {
    name: 'Telco Customer Churn',
    source: 'Kaggle',
    rows: 7043,
    columns: 21,
    target: 'Churn',
    challenge: 'Predict whether a customer will churn (cancel their subscription) based on account and demographic data.',
    columnTypes: [
        { name: 'CustomerId', type: 'categorical' },
        { name: 'Tenure', type: 'numeric' },
        { name: 'MonthlyCharges', type: 'numeric' },
        { name: 'TotalCharges', type: 'numeric' },
        { name: 'Contract', type: 'categorical' },
        { name: 'PaymentMethod', type: 'categorical' },
        { name: 'Churn', type: 'target' },
    ],
    sampleRows: [
        { CustomerId: '7590-VHVEG', Tenure: 1, MonthlyCharges: 29.85, TotalCharges: 29.85, Contract: 'Month-to-month', PaymentMethod: 'Electronic check', Churn: 'No' },
        { CustomerId: '5575-GNVDE', Tenure: 34, MonthlyCharges: 56.95, TotalCharges: 1889.50, Contract: 'One year', PaymentMethod: 'Mailed check', Churn: 'No' },
        { CustomerId: '3668-QPYBK', Tenure: 2, MonthlyCharges: 53.85, TotalCharges: 108.15, Contract: 'Month-to-month', PaymentMethod: 'Mailed check', Churn: 'Yes' },
        { CustomerId: '7795-CFOCW', Tenure: 45, MonthlyCharges: 42.30, TotalCharges: 1840.75, Contract: 'Two year', PaymentMethod: 'Bank transfer', Churn: 'No' },
        { CustomerId: '9237-HQITU', Tenure: 2, MonthlyCharges: 70.70, TotalCharges: 151.65, Contract: 'Month-to-month', PaymentMethod: 'Electronic check', Churn: 'Yes' },
        { CustomerId: '9305-CDSKC', Tenure: 8, MonthlyCharges: 99.65, TotalCharges: 820.50, Contract: 'Month-to-month', PaymentMethod: 'Electronic check', Churn: 'Yes' },
        { CustomerId: '1452-KIOVK', Tenure: 22, MonthlyCharges: 89.10, TotalCharges: 1949.40, Contract: 'Month-to-month', PaymentMethod: 'Electronic check', Churn: 'No' },
        { CustomerId: '6713-OKOMC', Tenure: 10, MonthlyCharges: 29.75, TotalCharges: 301.90, Contract: 'Month-to-month', PaymentMethod: 'Mailed check', Churn: 'No' },
    ],
};
const salesData = {
    name: 'Store Sales - Time Series Forecasting',
    source: 'Kaggle',
    rows: 1110375,
    columns: 9,
    target: 'Sales',
    challenge: 'Forecast daily store sales across multiple retail stores using time series and promotional data.',
    columnTypes: [
        { name: 'Store', type: 'numeric' },
        { name: 'DayOfWeek', type: 'numeric' },
        { name: 'Date', type: 'categorical' },
        { name: 'Sales', type: 'target' },
        { name: 'Customers', type: 'numeric' },
        { name: 'Promo', type: 'numeric' },
        { name: 'StateHoliday', type: 'categorical' },
        { name: 'SchoolHoliday', type: 'numeric' },
    ],
    sampleRows: [
        { Store: 1, DayOfWeek: 5, Date: '2015-07-31', Sales: 5263, Customers: 555, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 2, DayOfWeek: 5, Date: '2015-07-31', Sales: 5020, Customers: 524, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 3, DayOfWeek: 5, Date: '2015-07-31', Sales: 6855, Customers: 710, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 4, DayOfWeek: 5, Date: '2015-07-31', Sales: 6162, Customers: 624, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 5, DayOfWeek: 5, Date: '2015-07-31', Sales: 3195, Customers: 373, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 6, DayOfWeek: 5, Date: '2015-07-31', Sales: 4607, Customers: 485, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 7, DayOfWeek: 5, Date: '2015-07-31', Sales: 7572, Customers: 759, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
        { Store: 8, DayOfWeek: 5, Date: '2015-07-31', Sales: 5573, Customers: 568, Promo: 1, StateHoliday: '0', SchoolHoliday: 1 },
    ],
};
const digitData = {
    name: 'Digit Recognizer - MNIST',
    source: 'Kaggle',
    rows: 42000,
    columns: 785,
    target: 'Label',
    challenge: 'Classify handwritten digits (0-9) from 28x28 pixel grayscale images.',
    columnTypes: [
        { name: 'Pixel1', type: 'numeric' },
        { name: 'Pixel2', type: 'numeric' },
        { name: 'Pixel3', type: 'numeric' },
        { name: 'Pixel4', type: 'numeric' },
        { name: 'Pixel5', type: 'numeric' },
        { name: 'Label', type: 'target' },
    ],
    sampleRows: [
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 1 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 0 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 1 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 4 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 0 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 9 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 2 },
        { Pixel1: 0, Pixel2: 0, Pixel3: 0, Pixel4: 0, Pixel5: 0, Label: 1 },
    ],
};
const tweetsData = {
    name: 'Natural Language Processing with Disaster Tweets',
    source: 'Kaggle',
    rows: 7613,
    columns: 5,
    target: 'target',
    challenge: 'Classify whether a tweet is about a real disaster (1) or not (0) using NLP techniques.',
    columnTypes: [
        { name: 'id', type: 'numeric' },
        { name: 'keyword', type: 'categorical' },
        { name: 'location', type: 'categorical' },
        { name: 'text', type: 'categorical' },
        { name: 'target', type: 'target' },
    ],
    sampleRows: [
        { id: 1, keyword: 'ablaze', location: 'USA', text: 'Forest fire near downtown San Francisco', target: 1 },
        { id: 2, keyword: 'accident', location: 'London', text: 'Car crash on M5 near junction 23', target: 1 },
        { id: 3, keyword: 'earthquake', location: 'Japan', text: 'Magnitude 6.2 quake hits northern Japan', target: 1 },
        { id: 4, keyword: 'flood', location: 'India', text: 'Heavy rains cause flooding in Mumbai', target: 1 },
        { id: 5, keyword: 'storm', location: 'Florida', text: 'Tropical storm heading toward coast', target: 1 },
        { id: 6, keyword: 'fire', location: 'Australia', text: 'Bushfire threatens communities in NSW', target: 1 },
        { id: 7, keyword: 'rain', location: 'UK', text: 'Just another rainy day in London', target: 0 },
        { id: 8, keyword: 'weather', location: 'New York', text: 'Beautiful sunny day in Central Park', target: 0 },
    ],
};
const loanData = {
    name: 'Loan Default Prediction',
    source: 'Kaggle',
    rows: 255347,
    columns: 18,
    target: 'Default',
    challenge: 'Predict whether a borrower will default on their loan based on financial and demographic attributes.',
    columnTypes: [
        { name: 'LoanID', type: 'categorical' },
        { name: 'LoanAmount', type: 'numeric' },
        { name: 'InterestRate', type: 'numeric' },
        { name: 'Term', type: 'numeric' },
        { name: 'CreditScore', type: 'numeric' },
        { name: 'AnnualIncome', type: 'numeric' },
        { name: 'Default', type: 'target' },
    ],
    sampleRows: [
        { LoanID: 'LN001', LoanAmount: 15000, InterestRate: 8.5, Term: 36, CreditScore: 720, AnnualIncome: 65000, Default: 0 },
        { LoanID: 'LN002', LoanAmount: 25000, InterestRate: 11.2, Term: 60, CreditScore: 640, AnnualIncome: 48000, Default: 1 },
        { LoanID: 'LN003', LoanAmount: 10000, InterestRate: 6.8, Term: 24, CreditScore: 780, AnnualIncome: 85000, Default: 0 },
        { LoanID: 'LN004', LoanAmount: 35000, InterestRate: 14.5, Term: 72, CreditScore: 580, AnnualIncome: 42000, Default: 1 },
        { LoanID: 'LN005', LoanAmount: 20000, InterestRate: 9.1, Term: 48, CreditScore: 690, AnnualIncome: 55000, Default: 0 },
        { LoanID: 'LN006', LoanAmount: 5000, InterestRate: 5.5, Term: 12, CreditScore: 810, AnnualIncome: 95000, Default: 0 },
        { LoanID: 'LN007', LoanAmount: 40000, InterestRate: 16.0, Term: 84, CreditScore: 540, AnnualIncome: 38000, Default: 1 },
        { LoanID: 'LN008', LoanAmount: 18000, InterestRate: 7.9, Term: 36, CreditScore: 750, AnnualIncome: 72000, Default: 0 },
    ],
};
const edaData = {
    name: 'Iris Dataset - Exploratory Data Analysis',
    source: 'UCI / Kaggle',
    rows: 150,
    columns: 5,
    target: 'species',
    challenge: 'Explore, visualize, and summarize the classic Iris dataset to understand feature distributions and species separation.',
    columnTypes: [
        { name: 'sepal_length', type: 'numeric' },
        { name: 'sepal_width', type: 'numeric' },
        { name: 'petal_length', type: 'numeric' },
        { name: 'petal_width', type: 'numeric' },
        { name: 'species', type: 'target' },
    ],
    sampleRows: [
        { sepal_length: 5.1, sepal_width: 3.5, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
        { sepal_length: 4.9, sepal_width: 3.0, petal_length: 1.4, petal_width: 0.2, species: 'setosa' },
        { sepal_length: 4.7, sepal_width: 3.2, petal_length: 1.3, petal_width: 0.2, species: 'setosa' },
        { sepal_length: 7.0, sepal_width: 3.2, petal_length: 4.7, petal_width: 1.4, species: 'versicolor' },
        { sepal_length: 6.4, sepal_width: 3.2, petal_length: 4.5, petal_width: 1.5, species: 'versicolor' },
        { sepal_length: 6.9, sepal_width: 3.1, petal_length: 4.9, petal_width: 1.5, species: 'versicolor' },
        { sepal_length: 6.3, sepal_width: 3.3, petal_length: 6.0, petal_width: 2.5, species: 'virginica' },
        { sepal_length: 5.8, sepal_width: 2.7, petal_length: 5.1, petal_width: 1.9, species: 'virginica' },
    ],
};
const datasetMap = {
    'kaggle-titanic-project': titanicData,
    'kaggle-house-prices-project': housePricesData,
    'kaggle-credit-card-fraud-project': fraudData,
    'kaggle-customer-churn-project': churnData,
    'kaggle-store-sales-forecasting-project': salesData,
    'kaggle-digit-recognizer-project': digitData,
    'kaggle-disaster-tweets-project': tweetsData,
    'kaggle-loan-default-risk-project': loanData,
    'kaggle-eda-project': edaData,
};
function formatCell(val) {
    if (val === undefined || val === null || (typeof val === 'number' && isNaN(val)))
        return 'NaN';
    if (typeof val === 'number') {
        if (Number.isInteger(val))
            return val.toString();
        return val.toFixed(2);
    }
    return String(val);
}
function getTypeColor(type) {
    switch (type) {
        case 'numeric': return 'text-blue-700 bg-blue-50 border-blue-200';
        case 'categorical': return 'text-purple-700 bg-purple-50 border-purple-200';
        case 'target': return 'text-[#B6532B] bg-orange-50 border-orange-200';
    }
}
function getColumnStyle(col, colTypes) {
    const ct = colTypes.find(c => c.name === col);
    if (ct?.type === 'target')
        return 'text-[#B6532B] font-bold';
    return 'text-[#2E251E]';
}
export const KaggleDataPreview = () => {
    const { activeModule } = useApp();
    const isKaggle = activeModule.category === 'Kaggle Real-World Projects';
    const config = useMemo(() => datasetMap[activeModule.id] ?? null, [activeModule.id]);
    if (!isKaggle) {
        return (_jsxs("div", { className: "flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]", children: [_jsx(Database, { className: "w-16 h-16 mb-4 text-[#CFC5B4]" }), _jsx("h3", { className: "text-xl font-bold text-[#2E251E] mb-2", children: "Conceptual Overview" }), _jsx("p", { children: "This module is designed for reading and reflection rather than interactive simulation. Explore the theory tab for a deep dive." })] }));
    }
    if (!config) {
        return (_jsxs("div", { className: "flex flex-col items-center justify-center h-full p-12 text-center text-[#6E6257]", children: [_jsx(Database, { className: "w-16 h-16 mb-4 text-[#CFC5B4]" }), _jsx("h3", { className: "text-xl font-bold text-[#2E251E] mb-2", children: "Dataset Preview" }), _jsx("p", { children: "Dataset information is not available for this module." })] }));
    }
    const columns = Object.keys(config.sampleRows[0]);
    return (_jsxs("div", { className: "w-full h-full overflow-y-auto p-4 space-y-4", children: [_jsxs("div", { className: "flex items-start gap-3", children: [_jsx("div", { className: "p-2.5 rounded-lg bg-[#B6532B] bg-opacity-10", children: _jsx(Table, { className: "w-5 h-5 text-[#B6532B]" }) }), _jsxs("div", { className: "flex-1 min-w-0", children: [_jsx("h3", { className: "text-lg font-bold text-[#2E251E] truncate", children: config.name }), _jsx("p", { className: "text-sm text-[#6E6257] mt-0.5", children: config.source })] })] }), _jsxs("div", { className: "flex flex-wrap gap-3", children: [_jsxs("div", { className: "flex-1 min-w-[140px] bg-white rounded-lg border border-[#E5DDD0] p-3", children: [_jsxs("div", { className: "flex items-center gap-1.5 text-xs text-[#6E6257] mb-1", children: [_jsx(Info, { className: "w-3.5 h-3.5" }), _jsx("span", { children: "Challenge" })] }), _jsx("p", { className: "text-sm text-[#2E251E] leading-snug", children: config.challenge })] }), _jsxs("div", { className: "flex gap-3 flex-wrap", children: [_jsxs("div", { className: "bg-white rounded-lg border border-[#E5DDD0] p-3 min-w-[100px]", children: [_jsx("div", { className: "text-xs text-[#6E6257] mb-0.5", children: "Rows" }), _jsx("div", { className: "text-lg font-bold text-[#2E251E] font-mono", children: config.rows.toLocaleString() })] }), _jsxs("div", { className: "bg-white rounded-lg border border-[#E5DDD0] p-3 min-w-[100px]", children: [_jsx("div", { className: "text-xs text-[#6E6257] mb-0.5", children: "Columns" }), _jsx("div", { className: "text-lg font-bold text-[#2E251E] font-mono", children: config.columns })] }), _jsxs("div", { className: "bg-white rounded-lg border border-[#E5DDD0] p-3 min-w-[120px]", children: [_jsx("div", { className: "text-xs text-[#6E6257] mb-0.5", children: "Target" }), _jsx("div", { className: "text-base font-bold text-[#B6532B] font-mono", children: config.target })] })] })] }), _jsxs("div", { className: "bg-white rounded-lg border border-[#E5DDD0] overflow-hidden", children: [_jsxs("div", { className: "px-3 py-2 border-b border-[#E5DDD0] bg-[#FAF6EE] flex items-center gap-2", children: [_jsx(BarChart3, { className: "w-4 h-4 text-[#6E6257]" }), _jsx("span", { className: "text-sm font-semibold text-[#2E251E]", children: "Column Types" })] }), _jsx("div", { className: "flex flex-wrap gap-1.5 p-3", children: config.columnTypes.map(ct => (_jsx("span", { className: `text-xs px-2 py-0.5 rounded-md border font-mono ${getTypeColor(ct.type)}`, children: ct.name }, ct.name))) })] }), _jsxs("div", { className: "bg-white rounded-lg border border-[#E5DDD0] overflow-hidden", children: [_jsx("div", { className: "px-3 py-2 border-b border-[#E5DDD0] bg-[#FAF6EE]", children: _jsxs("span", { className: "text-sm font-semibold text-[#2E251E]", children: ["Data Preview (", config.sampleRows.length, " rows)"] }) }), _jsx("div", { className: "overflow-x-auto", children: _jsx("div", { className: "inline-block min-w-full align-middle", children: _jsx("div", { className: "overflow-hidden", children: _jsxs("table", { className: "min-w-full divide-y divide-[#E5DDD0]", children: [_jsx("thead", { children: _jsx("tr", { className: "bg-[#FAF6EE]", children: columns.map(col => (_jsx("th", { className: `sticky top-0 px-3 py-2.5 text-left text-xs font-semibold uppercase tracking-wider ${getColumnStyle(col, config.columnTypes)}`, children: col }, col))) }) }), _jsx("tbody", { className: "divide-y divide-[#E5DDD0]", children: config.sampleRows.map((row, i) => (_jsx("tr", { className: i % 2 === 0 ? 'bg-white' : 'bg-[#FAF6EE]', children: columns.map(col => {
                                                    const ct = config.columnTypes.find(c => c.name === col);
                                                    return (_jsx("td", { className: `px-3 py-2 text-xs font-mono whitespace-nowrap ${ct?.type === 'target'
                                                            ? 'font-bold text-[#B6532B]'
                                                            : 'text-[#2E251E]'}`, children: formatCell(row[col]) }, col));
                                                }) }, i))) })] }) }) }) })] })] }));
};
