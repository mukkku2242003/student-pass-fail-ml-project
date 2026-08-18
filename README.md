# Student Pass/Fail Prediction ML Project

This is a beginner machine learning project that predicts whether a student will pass or fail based on study hours and attendance.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib

## Project Workflow

1. Read student data from CSV
2. Clean missing values
3. Remove duplicate records
4. Encode target labels
5. Split data into training and testing sets
6. Train Random Forest Classifier
7. Check accuracy and classification report
8. Save trained model using Joblib
9. Load saved model and predict new student result

## Features

- study_hours
- attendance

## Target

- result: pass or fail

## How to Run

First train the model:

```bash
python train_model.py
