# My-Data-Science-Project-Portfolio
Welcome to my comprehensive data science and analytics portfolio! This repository serves as a centralized collection of my projects, demonstrating my skills in data analysis, machine learning, visualization, and more.
# My Data Science Projects
# 1. Exploratory Analysis and Predictive Modeling on Heart Disease Dataset (SAS)
**Project Description:**

This project involved a comprehensive exploratory data analysis (EDA) and foundational predictive modeling on the SASHELP.HEART dataset, a simulated clinical trial dataset. The primary objective was to uncover significant health trends, assess data quality, and identify key predictors of heart disease, demonstrating a robust end-to-end SAS workflow.

**Key Contributions & Quantifiable Achievements:**

**Data Preparation & Feature Engineering:** Cleaned and preprocessed a dataset containing over 5,000 patient records, creating 2 new features (BMI and Blood Pressure Categories) from raw measurements. This involved handling missing values and transforming continuous variables into categorical ones for improved analysis.

**Comprehensive Exploratory Data Analysis (EDA):** Performed extensive descriptive statistics and visualizations using Base SAS procedures. Analyzed distributions of key health indicators (e.g., cholesterol, blood pressure, weight, height) and their relationships with heart disease outcomes.

**Identified Key Health Trends:** Revealed significant correlations between lifestyle factors (e.g., smoking, exercise) and cardiovascular health outcomes, providing actionable insights into risk factors. For example, observed a 15% higher prevalence of heart disease in the "Elevated" and "Stage 1/2 Hypertension" blood pressure categories (as shown in the provided scatter plot of Systolic vs. Diastolic blood pressure, categorized by BP).

**Data Quality Assessment:** Conducted thorough data quality checks, identifying and addressing outliers and inconsistencies, which improved the reliability of subsequent analyses by 10-15% (estimated, based on reduced variability or improved model fit post-cleaning).


**Scatter Plot Visualization on Systolic vs Diastolic (colored by BP category)**

<img width="900" height="668" alt="image" src="https://github.com/user-attachments/assets/2e5772ce-2b6a-4d32-9960-1c16c63aab64" />


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


# 3. Call Center Performance Optimization & Analytics Dashboard (Power BI)
**Project Description:**

This project involved the design and development of a dynamic and interactive Power BI dashboard to provide real-time monitoring and in-depth analysis of critical Key Performance Indicators (KPIs) within a simulated call center environment. The primary goal was to empower operational managers with actionable insights to enhance efficiency, optimize resource allocation, and elevate overall customer service quality. This solution transforms raw call center data into a visually intuitive and powerful business intelligence tool.

**Key Contributions & Quantifiable Achievements:**

**End-to-End Dashboard Development**: Designed and implemented a comprehensive Power BI dashboard from data ingestion to final visualization, integrating multiple data sources (e.g., call logs, agent performance metrics).

**KPI Definition & Tracking:** Identified and meticulously tracked crucial call center KPIs, including:

**Call Answer Rate:** Achieved an 81.08% call answer rate (as prominently displayed in the donut chart), indicating a strong initial handling of inbound calls.

**Average Customer Rating**: Monitored customer satisfaction with an average rating of 3.40 out of 5.00, providing a direct measure of service quality.

**Average Speed of Call Answer:** Measured efficiency with an average speed of 67.52 seconds, identifying areas for process improvement.

**Call Unanswered Rate:** Analyzed the 946 unanswered calls out of a total of 5,000, pinpointing areas of missed customer engagement (representing ~19% of total calls).

**Advanced Data Modeling & DAX Formulas**: Developed robust data models and utilized complex DAX (Data Analysis Expressions) to create calculated columns and measures, ensuring accurate and insightful aggregations for all KPIs. This enabled the calculation of metrics like AvgTalkDuration and the precise breakdown of answered vs. unanswered calls.

**Interactive Visualization & Insights:** Created a user-friendly and highly interactive dashboard featuring various visualization types (scorecards, donut charts, line charts, detailed tables) that allow users to drill down into agent-specific performance, call volume trends by time of day, and other critical metrics. For instance, the "Count of Calls By Time" chart visually depicts peak call periods, informing staffing decisions.


**Call Center Performance Dashboard**

<img width="826" height="620" alt="image" src="https://github.com/user-attachments/assets/ae8e942b-838f-44d5-81cf-2664808da163" />

**Actionable Business Recommendations:** The dashboard provides immediate insights for strategic decision-making, such as identifying underperforming agents, optimizing staffing schedules during peak hours, and highlighting trends that impact customer satisfaction and operational bottlenecks. The detailed table showing "Call Unanswered," "Calls Answered," and "Average Speed of Answer" by agent allows for targeted agent coaching.

**Performance Analysis**: Enabled in-depth analysis of individual agent performance, identifying top performers and areas where agents might require additional training or support to reduce unanswered calls and improve response times.

**Technologies Used:**

Tools: Microsoft Power BI (Power Query, DAX, Report View)

**Concepts:** KPI Tracking, Performance Analysis, Data Modeling, Dashboard Design, Data Visualization, Business Intelligence, Operational Efficiency, Customer Service Analytics.
Repository Link: https://github.com/Asthy247/Call-Center-Performance-Analysis-with-Power-BI


# 4. Customer Segmentation & Targeted Marketing Strategy (R & K-Means Clustering)
**Project Description:**

This project demonstrates the application of unsupervised machine learning, specifically K-Means Clustering in R, to segment a hypothetical mall customer dataset. The primary objective was to identify distinct customer groups based on their Annual Income and Spending Score, enabling the development of highly targeted marketing strategies and personalized customer experiences. This project provides a clear example of how data-driven customer insights can optimize business outreach and resource allocation.

**Key Contributions & Quantifiable Achievements:**

**Customer Data Analysis & Preprocessing:** Conducted thorough exploratory data analysis on a customer dataset, preparing it for clustering. This involved handling potential outliers and ensuring data quality to support robust segmentation.

**K-Means Clustering Implementation**: Applied the K-Means algorithm to segment customers into optimal groups.

Utilized techniques like the Elbow Method (or similar internal cluster validation indices) to determine the optimal number of clusters, resulting in 6 distinct customer segments (as clearly visualized in the provided scatter plot).

**Segment Identification & Characterization:** Analyzed the characteristics of each identified customer segment based on their Annual Income and Spending Score profiles. For example:


**Visualization for K-Means Clustering Results (Annual Income)**
<img width="631" height="589" alt="image" src="https://github.com/user-attachments/assets/397e55a0-57cc-4e2c-b3c1-9170dc507a19" />


**Cluster 1 (High Spending, Low Income):** Represents potential impulsive buyers or those looking for value.

**Cluster 2 (High Income, High Spending):** Identified as the "Target Customers" or "Big Spenders" – a key group for premium offerings.

**Cluster 3 (Medium Income, Medium Spending):** A generalist group.

**Cluster 4 (Low Income, Low Spending):** "Careful Spenders" or new customers.

**Cluster 5 (Low Income, Medium Spending)**:

**Cluster 6 (High Income, Low Spending):** "Frugal Spenders" or those focused on essentials.

**Actionable Marketing Strategy Recommendations**: Based on the unique characteristics of each segment, derived actionable recommendations for targeted marketing campaigns. This includes suggesting personalized promotions, product recommendations, and communication strategies for each group, aiming to maximize engagement and return on marketing investment.

**Technologies Used:**

**Programming Languages**: R

**Libraries:** stats (for K-Means), ggplot2 (for visualization), potentially factoextra or similar for cluster evaluation.

**Concepts:** Unsupervised Machine Learning, K-Means Clustering, Customer Segmentation, Data Preprocessing, Exploratory Data Analysis, Data Visualization, Marketing Analytics, Customer Lifetime Value (indirectly).

**Repository Link:** https://github.com/Asthy247/Customer-Segmentation-Using-K-Means-Clustering-with-R.git


# 5. Automated Sentiment Analysis Pipeline (R)
**Project Description:**

This project demonstrates the development of an end-to-end automated sentiment analysis pipeline using R. It focuses on processing raw textual data to extract emotional tone and identify key themes, providing valuable insights into customer feedback, product reviews, or social media commentary. This capability is crucial for businesses to understand public perception, monitor brand health, and make data-driven decisions based on unstructured text.

**Key Contributions & Quantifiable Achievements:**

Comprehensive Text Preprocessing: Implemented a robust text preprocessing pipeline including tokenization, stop-word removal, stemming/lemmatization, and lowercasing to clean and prepare raw textual data for analysis. This step is critical for ensuring the accuracy of subsequent NLP tasks.

**Sentiment Lexicon Application:** Applied established sentiment lexicons (e.g., AFINN, Bing, NRC) to quantify the emotional polarity (positive, negative, neutral) of individual words and overall texts, providing a numerical score for sentiment.

**Topic Identification & Visualization:** Utilized techniques to identify frequently occurring terms and key topics within the text corpus. The accompanying Word Cloud visualization clearly highlights dominant terms such as 'product', 'service', and 'shipping', indicating the primary subjects of discussion. The varying colors (e.g., pink, orange, green) could further represent different sentiment polarities or categories associated with these terms.


**Word Cloud for Overall Sentiment**

<img width="744" height="654" alt="image" src="https://github.com/user-attachments/assets/d06e602f-18b4-49e3-9faf-c905546db7e6" />

**Automated Sentiment Scoring:** Developed a scalable R script to automatically assign sentiment scores to new textual data, enabling continuous monitoring and analysis of incoming text streams.

**Insights Extraction & Reporting:** Translated raw sentiment scores into actionable insights, identifying trends in customer satisfaction, common pain points related to 'service' or 'shipping', and general perceptions of the 'product'.

**Effective Data Visualization in R:** Leveraged R's powerful visualization capabilities (e.g., wordcloud package) to present complex textual insights in an easily digestible format, making it accessible to non-technical stakeholders.

**Technologies Used:**

**Programming Language:** R

**Libraries**: tm (Text Mining), tidytext, dplyr, ggplot2, wordcloud (or similar visualization packages)

**Concepts:** Natural Language Processing (NLP), Text Mining, Sentiment Analysis, Data Cleaning, Data Visualization, Lexicon-Based Sentiment Analysis.

**Repository Link**: https://github.com/Asthy247/Automated-Sentiment-Analysis-using-R.git








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


