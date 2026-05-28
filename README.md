# Heart Disease Prediction Application

## Overview

This project is an interactive machine learning application designed to assess the risk of heart disease based on 13 standard clinical parameters. Built with Python and Streamlit, the application trains a Logistic Regression model on historical patient data and provides real-time risk predictions and probability scores for new patient profiles.

## Features

* **Interactive User Interface:** A structured, clinical intake form built with Streamlit for seamless data entry.
* **Real-Time Model Training:** The application dynamically loads the dataset, processes features, and trains the machine learning model upon initialization.
* **Risk Assessment:** Outputs a binary classification (High Risk vs. Low Risk) alongside a specific probability percentage.
* **Model Evaluation:** Displays the baseline accuracy of the model using a standard 80/20 train-test split.

## Technology Stack

* **Language:** Python 3.x
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Logistic Regression, Train-Test Split, Accuracy Metrics)
* **Data Manipulation:** Pandas

## Dataset

The model relies on a standard heart disease dataset (e.g., UCI Cleveland Heart Disease dataset format) containing 13 clinical features and one target variable:

1. **Age:** Patient age in years
2. **Sex:** Male (1) or Female (0)
3. **Chest Pain Type (CP):** 4 distinct categories (Typical Angina, Atypical, Non-Anginal, Asymptomatic)
4. **Resting Blood Pressure (BP):** Measured in mm Hg
5. **Cholesterol (Chol):** Serum cholesterol in mg/dl
6. **Fasting Blood Sugar (FBS):** Indicator if > 120 mg/dl
7. **Resting ECG Results (RestECG):** Normal, ST-T wave abnormality, or left ventricular hypertrophy
8. **Maximum Heart Rate (MaxHR):** Maximum heart rate achieved during exercise
9. **Exercise Induced Angina (ExAng):** Yes (1) or No (0)
10. **ST Depression (Oldpeak):** Induced by exercise relative to rest
11. **Slope:** The slope of the peak exercise ST segment
12. **Major Vessels (CA):** Number of major vessels (0-3) colored by fluoroscopy
13. **Thalassemia (Thal):** Normal, fixed defect, or reversible defect

* **Target:** Presence of heart disease (1) or absence (0)

## Installation and Setup

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd <your-repository-directory>

```

**2. Install dependencies**
Ensure you have Python installed, then install the required packages:

```bash
pip install streamlit pandas scikit-learn

```

**3. Add the dataset**
Place your dataset file named `Heart_Disease_Prediction.csv` in the root directory of the project.

**4. Run the application**

```bash
streamlit run app.py

```

## Usage

Upon launching the application, the sidebar will display the current training size and model accuracy. Enter the specific vitals and test results for a patient into the three categorized columns (Personal Info, Vitals & Tests, Advanced Heart Checks). Click "Analyze Full Risk Profile" to generate the prediction.

## Future Enhancements

* Implement data preprocessing pipelines (e.g., scaling numerical features like Cholesterol and MaxHR) to improve model performance.
* Explore ensemble models like Random Forest or Gradient Boosting to compare baseline accuracies.
* Add data visualization charts to show feature importance and patient data distribution.
