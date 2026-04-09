# 🌍 Real-Time Air Quality Index (AQI) Prediction & Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-orange.svg)
![Data Science](https://img.shields.io/badge/Data%20Science-EDA%20%26%20Prediction-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
4. [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
5. [Machine Learning Model](#-machine-learning-model)
6. [Key Insights & Feature Importance](#-key-insights--feature-importance)
7. [Tools & Technologies](#-tools--technologies)
8. [How to Run the Project](#-how-to-run-the-project)
9. [Future Work](#-future-work)

---

## 🚀 Project Overview
This project analyzes real-time air quality data from various monitoring stations across India to predict the Air Quality Index (AQI) categories using a **Random Forest Classifier**. 

Beyond basic prediction, this project focuses heavily on **model interpretability**. By engineering new features and extracting "Feature Importance," the project uncovers the critical drivers of air pollution, revealing that *pollutant fluctuation (spread)* and *exact geographic coordinates* are far stronger indicators of air quality than broad regional data like city or state names.

---

## 📊 Dataset Description
The dataset contains real-time Air Quality Index (AQI) readings from various stations. 
* **Original Shape:** 3,207 rows × 11 columns
* **Cleaned Shape:** 3,043 rows × 12 columns
* **Key Features:**
  * `country`, `state`, `city`, `station`: Location metadata.
  * `latitude`, `longitude`: Precise geographic coordinates.
  * `last_update`: Timestamp of the reading.
  * `pollutant_id`: Type of pollutant (PM2.5, PM10, NO2, NH3, SO2, CO, OZONE).
  * `pollutant_min`, `pollutant_max`, `pollutant_avg`: Concentration metrics.

---

## 🛠 Data Preprocessing & Feature Engineering
To prepare the data for machine learning, a rigorous cleaning and feature engineering pipeline was implemented:

1. **Handling Missing & Faulty Data:** * Identified and flagged 163 rows where sensors were offline (missing `pollutant_avg`).
   * Removed anomalous rows where the average pollutant reading was mathematically impossible (lower than the minimum).
2. **Text Standardization:** Cleaned categorical strings (e.g., replacing underscores with spaces in state names and using `.title()`).
3. **Temporal Feature Extraction:** Converted `last_update` to datetime objects and extracted `hour`, `day`, and `month` to capture seasonal and daily pollution patterns.
4. **Feature Engineering - "The Spread":** Created a new feature called `spread` (`pollutant_max` - `pollutant_min`). This measures the stability or volatility of the air quality at a given station.
5. **Target Variable Creation:** Engineered the target variable `aqi_category` based on the `pollutant_avg`:
   * **Excellent:** <= 50
   * **Moderate:** 51 - 100
   * **Poor:** 101 - 200
   * **Severe:** > 200

---

## 📈 Exploratory Data Analysis (EDA)
Several visualizations were generated to understand the underlying data distributions:
* **The Extremes:** A horizontal bar chart comparing the Top 10 Cleanest vs. Top 10 Most Polluted cities for PM2.5.
* **State-Level Heatmaps:** A detailed report card highlighting the severity of different pollutants across the top 15 most polluted states.
* **Proportional Breakdowns:** Stacked bar charts illustrating the percentage of AQI categories in heavily monitored states.
* **Multidimensional Scatter Matrix:** Evaluated the statistical distributions and correlations of numerical features to prove class separability prior to modeling.

*(Note: PM2.5 was analyzed closely as it is a primary driver of overall AQI and a reliable warning source for other high pollutants).*

---

## 🤖 Machine Learning Model
The project utilizes a **Random Forest Classifier** to predict the `aqi_category`. This algorithm was chosen for its robustness to outliers, non-linear capabilities, and built-in feature importance tracking.

**Data Preparation:**
* **Encoding:** Categorical variables (`country`, `state`, `city`, `pollutant_id`) were transformed using `LabelEncoder`.
* **Scaling:** Features were normalized using `StandardScaler` *after* the train-test split (80/20) to completely prevent data leakage.

**Hyperparameters Used:**
```python
RandomForestClassifier(
    n_estimators=200,      # More trees for better learning stability
    max_depth=10,          # Prevent overfitting
    min_samples_split=5,   # Force better decision boundaries
    random_state=42,
    class_weight='balanced' # Handle class imbalances (e.g., fewer 'Severe' cases)
)
