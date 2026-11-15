# 🚚 APL Logistics – Machine Learning-Based Late Delivery Risk Prediction

### An End-to-End Machine Learning Project for Predicting Delivery Delays in Global Supply Chain Operations

<p align="center">

**Machine Learning | Python | Scikit-learn | XGBoost | Streamlit | Hugging Face | GitHub**

</p>

---

## 📌 Project Overview

Late deliveries are a major challenge in global supply chain and logistics operations. They can affect customer satisfaction, increase operational costs, disrupt inventory planning, and impact overall business performance.

This project develops a **Machine Learning-Based Late Delivery Risk Prediction System** for APL Logistics. It uses historical logistics and order-related data to estimate the probability of late delivery and categorize orders into different risk levels.

The system provides an interactive Streamlit dashboard where users can upload an order dataset, explore delivery risk patterns, analyze individual orders, identify high-risk shipments, and download an operational review list.

The trained machine learning model is hosted on Hugging Face, while the application is deployed using Streamlit Community Cloud and connected to GitHub.

## 🎯 Project Objectives

- Analyze historical logistics and order data to identify delivery risk patterns.
- Develop machine learning models to predict late-delivery risk.
- Compare Logistic Regression, Random Forest, and XGBoost algorithms.
- Evaluate models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- Identify the most influential features affecting predicted delivery risk.
- Build an interactive dashboard for order-level and operational risk analysis.
- Provide actionable insights to help logistics teams prioritize shipments requiring attention.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning and preprocessing |
| XGBoost | Gradient boosting model |
| Joblib | Model serialization and loading |
| Plotly | Interactive dashboard visualizations |
| Streamlit | Web application and dashboard |
| Google Colab | Model development and experimentation |
| Hugging Face | Hosting trained model artifacts |
| GitHub | Source code management and version control |
| Streamlit Community Cloud | Application deployment |

---

## 📂 Project Architecture

The project follows an end-to-end machine learning workflow:

1. **Data Collection:** Load the logistics dataset.
2. **Data Preprocessing:** Clean and prepare the dataset for machine learning.
3. **Exploratory Data Analysis:** Analyze delivery patterns, shipping modes, regions, and customer segments.
4. **Feature Engineering:** Prepare numerical and categorical features.
5. **Model Development:** Train three machine learning algorithms.
6. **Model Evaluation:** Compare predictive performance using multiple evaluation metrics.
7. **Model Selection:** Save the selected trained model and its feature information.
8. **Model Hosting:** Store the model artifacts on Hugging Face.
9. **Dashboard Development:** Build the interactive Streamlit application.
10. **Deployment:** Connect GitHub to Streamlit Community Cloud.

---

## 🤖 Machine Learning Models

Three machine learning algorithms were trained and evaluated.

### 1. Logistic Regression

A supervised classification algorithm used as a baseline model to estimate the probability of late delivery.

### 2. Random Forest Classifier

An ensemble learning algorithm that combines multiple decision trees to improve prediction performance and capture complex relationships in the data.

### 3. XGBoost Classifier

A gradient boosting algorithm that builds decision trees sequentially to improve predictive performance by learning from previous errors.

### 📊 Model Performance Comparison

The following are the reported results from the model comparison performed in Google Colab.

| Evaluation Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 69.49% | 69.95% | **70.09%** |
| Precision | 84.55% | **84.58%** | 83.75% |
| Recall | 54.26% | 55.27% | **56.38%** |
| F1-Score | 66.10% | 66.85% | **67.39%** |
| ROC-AUC | 0.7441 | **0.7753** | 0.7702 |

### Model Evaluation

- **Highest Accuracy:** XGBoost (70.09%).
- **Highest Precision:** Random Forest (84.58%).
- **Highest Recall:** XGBoost (56.38%).
- **Highest F1-Score:** XGBoost (67.39%).
- **Highest ROC-AUC:** Random Forest (0.7753).

The comparison demonstrates that different models perform differently across evaluation metrics. Recall and ROC-AUC are particularly useful when evaluating a system intended to identify potentially delayed deliveries.

*Note: These figures represent the reported model comparison results and should not be interpreted as guaranteed performance on future orders.*

---

## 📊 Interactive Streamlit Dashboard

The application provides four primary analytical modules and additional model insights.

### 1. Delay Risk Overview

Provides a summary of uploaded orders and their predicted delivery-risk distribution.

Dashboard metrics include:

- Total orders analyzed.
- High-risk orders.
- Medium-risk orders.
- Low-risk orders.
- Predicted risk distribution.
- Predicted late-delivery probability distribution.

### 2. Order-Level Risk Prediction

Allows users to select an individual order and examine:

- Predicted late-delivery probability.
- Predicted risk category.
- Shipping mode.
- Order region.
- Order country.
- Customer segment.
- Scheduled shipping days.
- Product category and market information.

### 3. Region & Shipping-Mode Risk Analysis

Analyzes predicted delivery risk across different operational dimensions.

Visualizations include:

- Average predicted risk by order region.
- Risk distribution by shipping mode.
- Average predicted risk by customer segment.

### 4. Operations Action Panel

Identifies orders requiring operational attention based on their predicted risk.

The dashboard provides:

- High-risk order queue.
- Predicted delivery probability.
- Shipping mode and region information.
- Customer segment details.
- Suggested operational responses.
- Downloadable high-risk order report.

### 5. Model Insights

Displays the top 15 predictive features influencing the model's predictions through a feature-importance visualization.

---

## 🔍 Important Predictive Features

The feature-importance analysis identifies the following influential features:

| Feature | Description |
|---|---|
| Days for shipment (scheduled) | Scheduled shipping duration |
| Shipping Mode – Standard Class | Standard shipping category |
| Shipping Mode – First Class | First-class shipping category |
| Shipping Mode – Second Class | Second-class shipping category |
| Shipping Mode – Same Day | Same-day shipping category |
| Latitude | Geographic coordinate |
| Longitude | Geographic coordinate |
| Type – TRANSFER | Order transaction type |
| Order Item Total | Total order item value |
| Order Item Discount | Discount applied |
| Order Item Discount Rate | Discount percentage |
| Sales | Sales value |
| Type – DEBIT | Transaction type |
| Product Price | Product price |
| Category ID | Product category identifier |

These features provide insights into the factors the trained model uses when estimating delivery risk.

---

## 📈 Example Dashboard Results

For the uploaded dataset containing 180,519 orders, the dashboard displayed the following predicted risk distribution:

| Risk Category | Number of Orders |
|---|---:|
| High Risk | 62,198 |
| Medium Risk | 108,292 |
| Low Risk | 10,029 |
| **Total** | **180,519** |

These values represent model-generated risk classifications for the uploaded dataset, not independently verified delivery outcomes.

---

## ⚙️ Model Hosting and Deployment Architecture

The project uses three platforms to separate development, model storage, and application deployment.

### Google Colab – Model Development

Google Colab was used for:

- Dataset preprocessing.
- Exploratory data analysis.
- Feature engineering.
- Model training.
- Model comparison.
- Feature-importance analysis.
- Saving trained model artifacts.

### Hugging Face – Model Storage

The trained model artifacts are hosted in the following Hugging Face repository:

**Repository:** `Vvg0007/apl-logistics-model`

Model files:

- `best_late_delivery_model.joblib`
- `feature_columns.joblib`

The Streamlit application downloads these artifacts when loading the prediction system.

### GitHub – Source Code Management

GitHub stores the application source code, dependency configuration, and project documentation.

**GitHub Repository:**

`Vikas-VGowda/APL-Logistics-Late-Delivery-Prediction`

### Streamlit Community Cloud – Application Deployment

Streamlit Community Cloud hosts the interactive dashboard and connects directly to the GitHub repository.

**Live Application:**

https://apl-logistics-late-delivery-prediction.streamlit.app/

---

## 📁 Repository Structure

```text
APL-Logistics-Late-Delivery-Prediction/
│
├── app.py
│   └── Main Streamlit dashboard application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── .gitignore
    └── Files excluded from Git tracking
```

The trained model artifacts are hosted separately on Hugging Face rather than being committed to GitHub.

---

## 💻 Installation and Local Setup

### Prerequisites

- Python 3.13
- Git
- A GitHub account
- A Hugging Face account

### Step 1: Clone the repository

```bash
git clone https://github.com/Vikas-VGowda/APL-Logistics-Late-Delivery-Prediction.git
```

### Step 2: Navigate to the project directory

```bash
cd APL-Logistics-Late-Delivery-Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv venv
```

Activate the environment:

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the application

```bash
streamlit run app.py
```

The application will open locally in your browser.

---

## 🚀 How to Use the Dashboard

1. Open the deployed Streamlit application.
2. Upload the logistics order dataset in CSV format using the sidebar.
3. Wait for the application to process the dataset and generate predictions.
4. Explore the Delay Risk Overview.
5. Select individual orders to inspect their predicted risk.
6. Analyze regional and shipping-mode risk patterns.
7. Review high-risk shipments in the Operations Action Panel.
8. Download the high-risk order queue for further operational analysis.
9. Explore the model's feature-importance visualization.

**Note:** The uploaded CSV must contain the input features expected by the trained preprocessing pipeline. Do not upload files containing confidential customer information to a public application.

---

## 🔐 Data Privacy and Security

- The original dataset is not bundled with the public GitHub repository.
- Trained model artifacts are stored separately on Hugging Face.
- Users should avoid uploading personally identifiable customer information.
- The application should be deployed with appropriate data-handling and access controls before production use.

---

## ⚠️ Limitations

- Predictions are based on patterns learned from historical training data.
- Predicted delivery risk does not guarantee that a shipment will be delayed.
- Model performance may change when applied to new geographical regions or operational conditions.
- Prediction quality depends on the quality and compatibility of the uploaded dataset.
- Feature importance indicates model influence, not necessarily causation.
- The reported evaluation metrics should be validated periodically against actual delivery outcomes.

---

## 🔮 Future Enhancements

Potential improvements include:

- Integration with real-time shipment tracking systems.
- Real-time delivery status monitoring.
- Advanced geospatial analysis of delivery patterns.
- Automated notifications for high-risk shipments.
- Model retraining using updated logistics data.
- Hyperparameter optimization for improved predictive performance.
- Integration with logistics management systems.
- More advanced explainable AI techniques for individual predictions.

---

## 👨‍💻 Project Information

**Project Title:** Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

**Domain:** Logistics and Supply Chain Analytics

**Project Category:** Supervised Machine Learning – Classification

**Application:** Interactive Predictive Analytics Dashboard

**Development Environment:** Google Colab

**Programming Language:** Python

**Deployment Platform:** Streamlit Community Cloud

**Model Hosting:** Hugging Face

**Source Code:** GitHub

---

## 🙌 Acknowledgements

This project was developed as part of a Data Science internship/project experience at Unified Mentor.

The project provided practical experience in data preprocessing, exploratory data analysis, machine learning model development, performance evaluation, feature-importance analysis, application development, and cloud deployment.

---

## ⭐ Conclusion

The APL Logistics Late Delivery Risk Prediction project demonstrates an end-to-end application of machine learning in logistics and supply chain operations.

By combining predictive modeling, data visualization, and an interactive web application, the system provides a way to identify potentially high-risk orders and prioritize operational attention.

The project demonstrates how machine learning can support data-driven decision-making and contribute to more proactive logistics management.

---

**If you find this project interesting, consider giving the repository a ⭐ on GitHub!**
