#  Heart Disease Prediction using Machine Learning

##  Project Overview
This project predicts the likelihood of cardiovascular disease using machine learning algorithms. The model analyzes patient medical data including vital signs, lifestyle factors, and clinical measurements to classify whether a patient has heart disease.

##  Objective
Develop a robust binary classification model to predict heart disease risk with high accuracy, helping in early detection and preventive healthcare.

##  Dataset
- **Source**: [Cardiovascular Disease Dataset - Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Records**: 70,000 patient records
- **Features**: 11 input features + 1 target variable
- **Type**: Binary Classification (0 = No Disease, 1 = Disease Present)

### Features Description:
| Feature | Description | Type |
|---------|-------------|------|
| age | Age in days (converted to years) | Numerical |
| gender | 1=Female, 2=Male | Categorical |
| height | Height in cm | Numerical |
| weight | Weight in kg | Numerical |
| ap_hi | Systolic blood pressure | Numerical |
| ap_lo | Diastolic blood pressure | Numerical |
| cholesterol | 1=normal, 2=above normal, 3=well above normal | Categorical |
| gluc | Glucose level: 1=normal, 2=above normal, 3=well above normal | Categorical |
| smoke | Smoking: 0=no, 1=yes | Binary |
| alco | Alcohol intake: 0=no, 1=yes | Binary |
| active | Physical activity: 0=no, 1=yes | Binary |
| **cardio** | **Target: 0=No disease, 1=Disease** | **Binary** |

##  Technologies Used
- **Python 3.8+**
- **Libraries**:
  - NumPy - Numerical computing
  - Pandas - Data manipulation
  - Matplotlib & Seaborn - Data visualization
  - Scikit-learn - Machine learning algorithms
  - Pickle - Model serialization

##  Project Workflow

### 1. Data Preprocessing
-  Removed duplicate records
-  Handled outliers in blood pressure, height, and weight
-  Created derived features (BMI, Mean Arterial Pressure, Pulse Pressure)
-  Converted age from days to years
-  Applied StandardScaler for feature normalization

### 2. Exploratory Data Analysis
- Target variable distribution analysis
- Feature correlation heatmap
- Outlier detection using box plots
- Feature vs target relationship visualization

### 3. Feature Engineering
- Created BMI (Body Mass Index)
- Calculated Mean Arterial Pressure (MAP)
- Computed Pulse Pressure
- Age group categorization

### 4. Model Training
Implemented and compared 7 machine learning algorithms:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Gradient Boosting

### 5. Model Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- ROC-AUC curve
- Feature importance analysis

##  Results

### Model Performance Comparison:
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 72.5% | 0.73 | 0.71 | 0.72 |
| Gradient Boosting | 72.1% | 0.72 | 0.70 | 0.71 |
| Logistic Regression | 71.8% | 0.71 | 0.69 | 0.70 |

**Best Model**: Random Forest Classifier with 72.5% accuracy

### Feature Importance (Top 5):
1. Systolic Blood Pressure (ap_hi)
2. Age
3. Weight
4. Cholesterol
5. BMI



##  Installation & Usage

### Prerequisites
Python 3.8 or higher
pip package manager

### Installation
Clone the repository
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction

Install dependencies
pip install -r requirements.txt

### Running the Project
Launch Jupyter Notebook
jupyter notebook

Open and run: notebooks/heart_disease_prediction.ipynb


### Making Predictions
import pickle
import numpy as np

Load model and scaler
with open('models/best_model_random_forest.pkl', 'rb') as f:
model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
scaler = pickle.load(f)

Example prediction
Input: [gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, age_years, bmi, map, pulse_pressure]
sample_data = np.array([[2, 170, 80, 130, 85, 2, 1, 0, 0, 1, 55, 27.68, 100, 45]])
sample_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_scaled)

print("Prediction:", "Heart Disease" if prediction == 1 else "No Heart Disease")



##  Visualizations
The project includes comprehensive visualizations:
- Distribution plots
- Correlation heatmaps
- Model performance comparisons
- Confusion matrices
- ROC curves
- Feature importance charts

##  Key Insights
1. **Age** is a strong predictor of heart disease risk
2. **Blood pressure** (especially systolic) significantly impacts prediction
3. **Lifestyle factors** (smoking, alcohol, physical activity) show moderate correlation
4. **BMI** and **weight** are important risk indicators
5. **Gender** shows slight variation in disease prevalence

## 🎓 Academic Context
This project was developed as part of a Machine Learning course assignment, demonstrating:
- Complete ML pipeline implementation
- Data preprocessing and cleaning techniques
- Feature engineering skills
- Model evaluation and comparison
- Real-world healthcare application

##  Author
Reem Hamraz

##  License
This project is open source and available under the MIT License.


