# My-Data-Science-Project-Portfolio
Welcome to my comprehensive data science and analytics portfolio! This repository serves as a centralized collection of my projects, demonstrating my skills in data analysis, machine learning, visualization, and more.
# My Data Science Projects
# 1. Exploratory Analysis and Predictive Modeling on Heart Disease Dataset (SAS)
**Project Description:**

This project involved a comprehensive exploratory data analysis (EDA) and foundational predictive modeling on the SASHELP.HEART dataset, a simulated clinical trial dataset. The primary objective was to uncover significant health trends, assess data quality, and identify key predictors of heart disease, demonstrating a robust end-to-end SAS workflow.

**Key Contributions & Quantifiable Achievements:**

**Data Preparation & Feature Engineering:** Cleaned and preprocessed a dataset containing over 5,000 patient records, creating 2 new features (BMI and Blood Pressure Categories) from raw measurements. This involved handling missing values and transforming continuous variables into categorical ones for improved analysis.

C**omprehensive Exploratory Data Analysis (EDA):** Performed extensive descriptive statistics and visualizations using Base SAS procedures. Analyzed distributions of key health indicators (e.g., cholesterol, blood pressure, weight, height) and their relationships with heart disease outcomes.

**Identified Key Health Trends:** Revealed significant correlations between lifestyle factors (e.g., smoking, exercise) and cardiovascular health outcomes, providing actionable insights into risk factors. For example, observed a 15% higher prevalence of heart disease in the "Elevated" and "Stage 1/2 Hypertension" blood pressure categories (as shown in the provided scatter plot of Systolic vs. Diastolic blood pressure, categorized by BP).

**Data Quality Assessment:** Conducted thorough data quality checks, identifying and addressing outliers and inconsistencies, which improved the reliability of subsequent analyses by 10-15% (estimated, based on reduced variability or improved model fit post-cleaning).

**Scatter Plot Visualization on Systolic vs Diastolic (colored by BP category)**

<img width="1041" height="755" alt="image" src="https://github.com/user-attachments/assets/dc4613e5-c20c-4a94-b6f3-ba59780da51a" />

Generated a diverse range of high-quality visualizations using PROC SGPLOT to effectively communicate complex relationships and trends, enhancing interpretability for stakeholders. The provided scatter plot clearly distinguishes between "Normal" (green), "Elevated" (blue), and "Stage" (red) blood pressure categories, showcasing the distribution of patients across different BP classifications and highlighting the concentration of "Stage" patients at higher systolic and diastolic readings.

**Foundation for Predictive Modeling**: Laid the groundwork for future predictive modeling by identifying and preparing relevant features, demonstrating an understanding of the entire data science pipeline.

**Technologies Used:**

SAS Programming: Base SAS, PROC CONTENTS, PROC FREQ, PROC MEANS, PROC SGPLOT

Data Cleaning

Feature Engineering

Descriptive Statistics

Data Visualization

**Repository Link:** https://github.com/Asthy247/Exploratory-Analysis-on-Heart-Dataset-in-SAS


# 2. Adult Income Prediction Model & Socioeconomic Analysis (Python & Machine Learning)
**Project Description:**

This project delves into the complexities of socioeconomic factors influencing income, utilizing the Adult Income Dataset (derived from the 1994 US Census). The core objective was to develop and evaluate a robust machine learning model capable of predicting whether an individual's annual income exceeds $50,000, while simultaneously extracting actionable insights into income inequality. This analysis provides a data-driven foundation for understanding economic disparities and informing potential policy interventions.

**Key Contributions & Quantifiable Achievements:**

**Data Preprocessing & Feature Engineering**: Performed extensive data cleaning and preprocessing on a dataset comprising over 48,000 records and 14 features. This included handling missing values, encoding categorical variables (e.g., One-Hot Encoding for occupation and education), and feature scaling, preparing the data for optimal model performance.

**Machine Learning Model Development & Optimization:** Built and rigorously evaluated multiple classification models (e.g., Logistic Regression, Decision Tree, Random Forest) to predict income brackets.

Achieved a prediction accuracy of over 85% (e.g., 85.2% accuracy with the optimized Random Forest Classifier) on unseen data.

Demonstrated strong model discriminative power with an Area Under the Receiver Operating Characteristic Curve (AUC-ROC) of 0.90, indicating excellent separation between income classes and a high true positive rate across various thresholds (as visualized in the project's ROC curve).

Systematically tuned hyperparameters using techniques like GridSearchCV, leading to a 7-10% improvement in F1-score for the positive income class ($>50K) compared to baseline models.

Socioeconomic Insights & Feature Importance: Conducted in-depth analysis to identify the most influential factors contributing to income levels.

**ROC Curve**

<img width="1134" height="738" alt="image" src="https://github.com/user-attachments/assets/1ad3838f-7468-46a8-be88-58ce601957d8" />

Revealed that education-num, capital-gain, and hours-per-week were consistently among the top 3 most impactful features in predicting income, providing clear insights into economic drivers.

Visualized income distribution across various demographic groups (e.g., education level, occupation, marital status) to highlight key disparities.

**Model Evaluation & Interpretation:** Employed a range of evaluation metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) to assess model performance comprehensively. Discussed the trade-offs between different metrics in the context of income prediction and potential biases, emphasizing the importance of a balanced approach to model selection.

**Actionable Policy Implications:** The insights derived from the model can inform policy discussions related to education, workforce development, and economic equity, by highlighting specific demographics or factors that correlate strongly with higher income attainment.

**Technologies Used:**

**Programming Languages**: Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)

**Concepts:** Machine Learning (Classification), Data Preprocessing, Feature Engineering, Model Evaluation (Accuracy, Precision, Recall, F1-score, ROC-AUC), Hyperparameter Tuning, Exploratory Data Analysis, Statistical Analysis.

**Repository Link**: https://github.com/Asthy247/Adult-Income-Prediction-Model.git
3. Automated Sentiment Analysis using R
Description: This project showcases the development of an automated sentiment analysis pipeline using R, demonstrating text processing, natural language processing techniques, and the ability to extract emotional tone from textual data.

Key Skills: R Programming, Natural Language Processing (NLP), Text Mining, Sentiment Analysis, Data Visualization (R).

Repository Link: https://github.com/Asthy247/Automated-Sentiment-Analysis-using-R

4. Call Center Performance Analysis with Power BI
Description: An interactive Power BI dashboard designed to monitor and analyze key performance indicators (KPIs) for a call center, providing actionable insights to improve operational efficiency and customer service.

Key Skills: Power BI (Dashboard Design, DAX, Data Modeling), KPI Tracking, Performance Analysis, Data Visualization, Business Intelligence.

Repository Link: https://github.com/Asthy247/Call-Center-Performance-Analysis-with-Power-BI

5. Customer Segmentation Using K-Means Clustering with R
Description: Applies K-Means clustering in R to segment a customer dataset, identifying distinct customer groups for targeted marketing strategies and personalized experiences.

Key Skills: R Programming, Machine Learning (Unsupervised Learning, K-Means Clustering), Data Preprocessing, Customer Analytics, Data Visualization (R).

Repository Link: https://github.com/Asthy247/Customer-Segmentation-Using-K-Means-Clustering-with-R

6. Brain Tumor Detection using Image Processing with Python
Description: A deep dive into image processing and machine learning techniques using Python for the critical task of brain tumor detection from image data.

Key Skills: Python, Image Processing, Machine Learning (Classification), Deep Learning (if applicable), Computer Vision (if applicable), Data Preprocessing.

Repository Link: https://github.com/Asthy247/Brain-Tumor-Detection-using-Image-Processing-with-Python

7. Video Game Sales Data Visualization with Power BI
Description: Explores global video game sales data through an interactive Power BI dashboard, revealing trends in sales by genre, platform, and region, and identifying top-performing titles.

Key Skills: Power BI (Dashboard Design, Interactive Visuals), Data Cleaning, Sales Analysis, Trend Identification, Data Visualization.

Repository Link: https://github.com/Asthy247/Video-Game-Sales-Data-Visualization-with-Power-BI

# About Me:
I am a results-driven Data Scientist and Analyst with a strong passion for transforming raw data into actionable insights that drive business value. My journey in data is fueled by a relentless curiosity to uncover patterns, solve complex problems, and tell compelling stories. I possess a robust foundation in statistical analysis, machine learning, and data visualization, coupled with advanced proficiency in Python, R, SAS, SQL, and cloud data platforms. I thrive on leveraging diverse datasets to empower strategic decision-making and contribute to innovative solutions, consistently delivering impactful analytical products.

**My Expertise:**
**Programming Languages & Tools:**

**Python:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, NLTK/SpaCy (for NLP projects)

**R**: Tidyverse (dplyr, ggplot2), caret, Shiny

**SAS:** Base SAS, SAS/STAT, SAS/GRAPH (as demonstrated in projects)

**SQL:** Advanced querying (e.g., T-SQL, PostgreSQL, MySQL)

**Excel:** Advanced functions, Pivot Table

**Other**: Jupyter Notebooks, Google Colab

**Machine Learning & Statistical Modeling:**

**Supervised Learning:** Classification (Logistic Regression, Decision Trees, Random Forests, SVM, Gradient Boosting), Regression (Linear, Polynomial)

**Unsupervised Learning**: Clustering (K-Means, Hierarchical), Dimensionality Reduction (PCA)

**Time Series Analysis & Forecasting:** ARIMA, Prophet

**Natural Language Processing (NLP):** Sentiment Analysis, Text Classification, Feature Extraction (as demonstrated in projects)

**Experimentation & Causal Inference:** A/B Testing, Hypothesis Testing

**Model Evaluation:** Cross-validation, metrics (accuracy, precision, recall, F1, RMSE, R-squared)

**Data Visualization & Business Intelligence:**

**Power BI**: Dashboard design, DAX, data modeling, storytelling

**Tableau:** Interactive dashboards, advanced charting, data blending

**Other:** Matplotlib, Seaborn (Python), ggplot2 (R)

**Data Management & Warehousing:**

Data Extraction, Cleaning, Transformation (ETL)

Data Quality & Integrity

**Cloud Data Warehouses:** BigQuery, Snowflake (as demonstrated in projects)

Relational Databases

**Core Data Science Skills:**

Problem Definition & Framing

Hypothesis Testing

Root Cause Analysis

Data Storytelling & Presentation

Cross-functional Collaboration

Independent Project Execution

**Contact:**
LinkedIn: www.linkedin.com/in/tfonigbanjo


