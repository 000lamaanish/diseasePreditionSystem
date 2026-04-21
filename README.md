"# diseasePreditionSystem" 

# 🧪 Diabetes Risk Prediction System

An AI-powered machine learning web application that predicts the risk of diabetes based on medical parameters.

---

## 🚀 Project Overview

This project uses a **Random Forest Classifier** trained on the Pima Indians Diabetes dataset to predict whether a person is at risk of diabetes.

The system is deployed using **Streamlit** and provides an interactive UI for users.

---

## 🧠 Features Used

- Pregnancies  
- Glucose level  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

---

## ⚙️ Machine Learning Pipeline

- Data Preprocessing  
- Standardization (StandardScaler)  
- Random Forest Classifier  
- Model evaluation using Accuracy, Precision, Recall  
- Threshold tuning for better recall  

---

## 📊 Model Performance

- Accuracy: ~75–80%  
- Recall optimized for diabetes detection (~90% in tuned models)  
- Feature importance visualization included  

---

## 🖥️ Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Streamlit  
- Matplotlib  

---

## ▶️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
````

### 2. Run Streamlit app

```bash
streamlit run app/app.py
```

---

## 📂 Project Structure

```
diabetes_project/
│
├── notebook/          # Model training notebook
├── app/               # Streamlit application
│   ├── app.py
│   └── model.pkl
├── README.md
```

---

## 🎯 Objective

To build a simple but effective **healthcare AI system** that demonstrates:

* Machine learning pipeline development
* Model deployment
* Real-world problem solving

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

---

## 👨‍💻 Author

Built as part of a personal machine learning learning journey.

```
```
