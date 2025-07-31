# British Airways
# Customer Booking Prediction using Random Forest

[![Status: Completed](https://img.shields.io/badge/Status-Completed-green.svg)](https://shields.io/)
[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Framework: Scikit-learn](https://img.shields.io/badge/Framework-Scikit--learn-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete end-to-end machine learning pipeline to predict customer bookings using structured airline data. This project implements robust data preprocessing, feature engineering, class balancing, model training, evaluation, and interpretability using a **Random Forest classifier**.

---

## 📖 Table of Contents
- [1. Business Problem & Objective](#1-business-problem--objective)
- [2. Dataset Description](#2-dataset-description)
- [3. Project Directory Structure](#3-project-directory-structure)
- [4. The Machine Learning Pipeline](#4-the-machine-learning-pipeline)
- [5. Technical Stack](#5-technical-stack)
- [6. Setup & Installation Guide](#6-setup--installation-guide)
- [7. Author](#7-author)

---

## 1. Business Problem & Objective

### Business Problem
For an airline like **British Airways**, understanding customer intent is crucial for optimizing operations and revenue. A significant challenge lies in distinguishing between customers who are merely Browse flight options and those who will proceed to a final booking. Inaccurately forecasting booking volumes can lead to inefficient resource allocation (e.g., seat management, crew scheduling) and missed revenue opportunities from dynamic pricing and targeted marketing.

### Project Objective
The goal of this project is to develop a reliable classification model that **predicts whether a customer will complete a booking**. By analyzing customer interaction data, the model will provide a probability of conversion for each session. This enables the business to:
- **Optimize Marketing Spend:** Target promotions and offers to customers with a high probability of churn.
- **Improve Dynamic Pricing:** Adjust fares based on predicted demand.
- **Enhance Resource Planning:** More accurately forecast passenger loads.

---

## 2. Dataset Description

The analysis uses the `customer_booking.csv` dataset, which contains anonymized records of customer interactions during the flight search process.

- **Source:** Internal transactional and weblog data (simulated).
- **Size:** Contains thousands of unique customer sessions.
- **Target Variable:** `booking_complete` - A binary flag where `1` indicates a completed booking and `0` indicates the customer did not book.
- **Key Features:**
    - `num_passengers`: Number of passengers in the booking session.
    - `sales_channel`: `Internet` or `Mobile`.
    - `trip_type`: `RoundTrip`, `OneWay`, or `CircleTrip`.
    - `purchase_lead`: Days between the search and the intended departure.
    - `flight_day`: Day of the week for the flight.
    - `booking_origin`: Customer's country of origin.
    - `wants_extra_baggage`, `wants_preferred_seat`, `wants_in_flight_meals`: Boolean flags for ancillaries.

---

## 3. Project Directory Structure

The repository is organized to ensure modularity and reproducibility.

```

British Airways Booking Prediction/
│
├── data/
│   └── customer\_booking.csv        \# Raw input data
│
├── outputs/
│   ├── plots\_phase2/               \# Plots related to initial data inspection
│   ├── plots\_phase3/               \# Plots for Exploratory Data Analysis (EDA)
│   ├── ...                         \# Other plot directories for different phases
│   └── plots\_phase8/
│
├── models/
│   └── random\_forest\_booking.pkl   \# The final, serialized machine learning model
│
├── logs/
│   └── pipeline.log                \# Log file for tracking pipeline execution and errors
│
└── notebook.ipynb                  \# Jupyter Notebook containing the end-to-end code

````

---

## 4. The Machine Learning Pipeline

The project is structured as a sequence of 10 distinct phases, from setup to deployment readiness.

### Phase 1: Environment Setup
- **Action:** Import libraries, configure global settings (random seed for reproducibility), and apply a consistent `Rose Pine` dark theme for all visualizations.
- **Rationale:** Establishing a consistent and reproducible environment is the first step in any professional data science project.

### Phase 2: Data Loading & Initial Inspection
- **Action:** Load the `customer_booking.csv` file, handling potential encoding errors. Display the first few records (`.head()`) and check for null values (`.isnull().sum()`).
- **Outcome:** The initial plot of the target variable `booking_complete` immediately reveals a significant **class imbalance**, with far more non-bookings than bookings. This insight is critical and directly informs the need for balancing techniques later.
- **Plot:**
  ![Booking Status Distribution](outputs/plots_phase2/booking_status_distribution.png)

### Phase 3: Exploratory Data Analysis (EDA)
- **Action:** Conduct a visual exploration of key features. This includes plotting a missing value heatmap, the top 10 booking origins, and the distribution of passenger counts.
- **Rationale:** EDA helps build intuition about the data. For instance, identifying top origins highlights key markets, while analyzing passenger counts reveals common travel group sizes.
- **Plots:**
  ![Booking Origins](outputs/plots_phase3/top10_booking_origins.png)
  ![Passenger Count](outputs/plots_phase3/passenger_count_distribution.png)

### Phase 4: Data Cleaning, Splitting & Balancing
- **Action:** Drop irrelevant columns, split the data into training (80%) and testing (20%) sets, and then apply **SMOTE (Synthetic Minority Over-sampling Technique)** to the *training data only*.
- **Rationale:**
    - **Splitting:** We split *before* any other preprocessing to prevent **data leakage**, ensuring the test set remains a truly unseen evaluation set.
    - **SMOTE:** This technique addresses class imbalance by generating synthetic samples of the minority class (bookings), preventing the model from becoming biased towards the majority class (non-bookings). It is applied only to the training data.

### Phase 5: Feature Engineering
- **Action:** Create a new feature, `lead_time`, by combining `purchase_lead` and `flight_day`. The original columns are then dropped.
- **Rationale:** `lead_time` is a more powerful predictor, capturing the total planning duration. Dropping the original columns prevents multicollinearity and data redundancy.

### Phase 6: Categorical Encoding
- **Action:** Apply **One-Hot Encoding** to all categorical features (`object` data type).
- **Rationale:** Machine learning models require numerical input. One-hot encoding transforms categories into a numerical format by creating new binary columns for each unique category value, allowing the model to interpret them without assuming an ordinal relationship.
- **Plot:**
  ![Unique Categorical Values](outputs/plots_phase6/categorical_unique_values.png)

### Phase 7: Model Validation Strategy
- **Action:** Perform **5-fold Stratified Cross-Validation** on the training data using the Random Forest classifier, evaluating it on the **ROC-AUC** metric.
- **Rationale:**
    - **Stratified K-Fold:** Ensures that the proportion of bookings vs. non-bookings is the same in each fold, which is critical for a reliable evaluation on imbalanced data.
    - **ROC-AUC Score:** This metric is well-suited for imbalanced classification as it measures the model's ability to distinguish between the positive and negative classes across all probability thresholds.

- **Plot:**
  ![ROC-AUC Cross-Validation](outputs/plots_phase7/roc_auc_cross_validation.png)

### Phase 8: Final Model Evaluation
- **Action:** Evaluate the final trained model on the held-out test set. Generate a detailed **Classification Report** (Precision, Recall, F1-Score) and a **Confusion Matrix**.
- **Rationale:** This step provides a definitive assessment of the model's real-world performance. The confusion matrix gives a clear visual breakdown of correct and incorrect predictions for each class.
- **Plot:**
  ![Confusion Matrix](outputs/plots_phase8/confusion_matrix.png)

### Phase 9: Model Persistence
- **Action:** Serialize and save the trained model object to a file (`random_forest_booking.pkl`) using `joblib`.
- **Rationale:** This allows the model to be easily loaded and used for inference in a production environment without needing to retrain it every time.

### Phase 10: Model Interpretability
- **Action:** Extract and visualize the top feature importances from the trained Random Forest model.
- **Rationale:** Understanding which features drive the model's predictions (e.g., `lead_time`, `sales_channel`) provides actionable insights for the business and builds trust in the model's decisions.

---

## 5. Technical Stack

- **Python 3.9+:** Core programming language.
- **Pandas:** For high-performance data manipulation and analysis.
- **Scikit-learn:** For implementing the Random Forest model, cross-validation, and evaluation metrics.
- **imbalanced-learn:** To perform SMOTE for handling class imbalance.
- **Matplotlib & Seaborn:** For data visualization and generating insightful plots.
- **Joblib:** For efficient serialization of the final model.

---

## 6. Setup & Installation Guide

To replicate this analysis, follow these steps:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/costaspinto/your-repo-name.git](https://github.com/costaspinto/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create and Activate a Virtual Environment** (Recommended)
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Analysis**
    Open and run the Jupyter Notebook `notebook.ipynb` cell by cell. All outputs, including plots and the final model, will be generated in their respective directories.

---

## 7. Author

**Costas Pinto**
* MCA Student | ML Engineer
* **LinkedIn:** [linkedin.com/in/costaspinto](https://www.linkedin.com/in/costaspinto)
* **GitHub:** [github.com/costaspinto](https://github.com/costaspinto)

