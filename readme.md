# Customer Booking Prediction using Random Forest

A complete end-to-end machine learning pipeline to predict customer bookings using structured airline data. This project implements robust data preprocessing, feature engineering, class balancing, model training, evaluation, and interpretability using a Random Forest classifier.

---

## Project Overview

This project demonstrates a real-world classification use case using a **Random Forest Classifier** to predict whether a customer will complete a booking based on their interaction data. The pipeline follows a modular and industry-standard structure with visualizations, logging, error handling, and theme consistency.

---

## Directory Structure

```
British Airways Booking Prediction
│
├── data/
│   └── customer_booking.csv
│
├── outputs/
│   ├── plots_phase2/
│   ├── plots_phase3/
│   ├── plots_phase6/
│   ├── plots_phase7/
│   └── plots_phase8/
│
├── models/
│   └── random_forest_booking.pkl
│
├── logs/
│   └── pipeline.log
│
└── notebook.ipynb
```

---

## Pipeline Phases

### Phase 1: Environment Setup

* Imports essential libraries
* Sets global random seed and aesthetics
* Applies Rose Pine dark plotting theme

---

### Phase 2: Data Loading & Inspection

* Loads CSV data using multiple encodings
* Checks null values and previews records
* **Plot:**
  ![Booking Status Distribution](outputs/plots_phase2/booking_status_distribution.png)

---

### Phase 3: Data Overview & Visualization

* Missing value heatmap
* Top 10 booking origins
* Passenger count distribution
* **Plots:**
  ![Booking Origins](outputs/plots_phase3/top10_booking_origins.png)
  ![Passenger Count](outputs/plots_phase3/passenger_count_distribution.png)

---

### Phase 4: Data Cleaning, Splitting & SMOTE

* Drops unused/missing fields
* Splits data into training and testing
* Balances dataset using SMOTE
* Visual confirmation of balanced target variable

---

### Phase 5: Feature Engineering

* Creates `leadtime` from purchase/flight dates
* Drops original redundant columns
* Confirms new feature distributions

---

### Phase 6: Categorical Encoding

* One-hot encodes all object/categorical columns
* Visualizes number of unique values
* **Plot:**
  ![Unique Categorical Values](outputs/plots_phase6/categorical_unique_values.png)

---

### Phase 7: Cross-Validation (ROC-AUC)

* Performs 5-fold stratified ROC-AUC cross-validation
* **Plot:**
  ![ROC-AUC Cross-Validation](outputs/plots_phase7/roc_auc_cross_validation.png)

---

### Phase 8: Model Evaluation on Test Set

* Accuracy, precision, recall, F1, classification report
* Confusion matrix heatmap
* **Plot:**
  ![Confusion Matrix](outputs/plots_phase8/confusion_matrix.png)

---

### Phase 9: Model Saving

* Serializes final model to disk with joblib
* Saved to `models/random_forest_booking.pkl`

---

### Phase 10: Feature Importance

* Extracts top features using `.feature_importances_`
* Supports model explainability and optimization

---

## Color Theme

All plots follow the consistent **Rose Pine dark theme** to ensure visual coherence and professional aesthetics.

---

## How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook `notebook.ipynb`
4. Inspect generated outputs in the `outputs/` folder

---

## Requirements

* Python 3.9+
* scikit-learn
* pandas
* matplotlib
* seaborn
* imbalanced-learn
* joblib

---

## Status

**Completed** — ready for deployment or integration into business decision systems.

---

## Author

**Costas Pinto**
*MCA Student | ML Engineer  
[LinkedIn](https://www.linkedin.com/in/costaspinto) | [GitHub](https://github.com/costaspinto)

---
