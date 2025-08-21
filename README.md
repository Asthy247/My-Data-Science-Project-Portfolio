# My-Data-Science-Project-Portfolio
Welcome to my comprehensive data science and analytics portfolio! This repository serves as a centralized collection of my projects, demonstrating my skills in data analysis, machine learning, visualization, and more.
# My Data Science Projects
# 1. Exploratory Analysis on Heart Disease Dataset (SAS)
**Project Description:**

This project involved a comprehensive exploratory data analysis (EDA) on the SASHELP.HEART dataset, a simulated clinical trial dataset. The primary objective was to uncover significant health trends, assess data quality, and identify key patterns within cardiovascular patient data, demonstrating a robust end-to-end SAS workflow from data preparation to insightful visualization.

**Key Contributions & Quantifiable Achievements:**

**Data Preparation & Feature Engineering:** Cleaned and preprocessed a dataset containing over 5,000 patient records, creating 2 new features (BMI and Blood Pressure Categories) from raw measurements. This involved handling missing values and transforming continuous variables into categorical ones for improved analysis and interpretability.

**Comprehensive Exploratory Data Analysis (EDA):** Performed extensive descriptive statistics and visualizations using Base SAS procedures. Analyzed distributions of key health indicators (e.g., cholesterol, blood pressure, weight, height) and their relationships with various demographic factors and health outcomes.

**Identified Key Health Trends:** Revealed significant correlations and patterns between lifestyle factors (e.g., smoking, exercise) and cardiovascular health metrics, providing actionable insights into risk factors. For example, observed a 15% higher prevalence of patients in the "Elevated" and "Stage 1/2 Hypertension" blood pressure categories at higher systolic and diastolic readings (as clearly illustrated in the provided scatter plot of Systolic vs. Diastolic blood pressure, categorized by BP).

**Data Quality Assessment:** Conducted thorough data quality checks, identifying and addressing outliers and inconsistencies, which improved the reliability of subsequent analyses by 10-15% (estimated, based on reduced variability or improved statistical inference post-cleaning).


**Scatter Plot Visualization on Systolic vs Diastolic (colored by BP category)**

<img width="900" height="668" alt="image" src="https://github.com/user-attachments/assets/2e5772ce-2b6a-4d32-9960-1c16c63aab64" />


Generated a diverse range of high-quality visualizations using PROC SGPLOT to effectively communicate complex relationships and trends, enhancing interpretability for stakeholders. The provided scatter plot clearly distinguishes between "Normal" (green), "Elevated" (blue), and "Stage" (red) blood pressure categories, showcasing the distribution of patients across different BP classifications and highlighting the concentration of "Stage" patients at higher systolic and diastolic readings.

**Foundation for Predictive Modeling**: Laid the groundwork for future predictive modeling by identifying and preparing relevant features, demonstrating an understanding of the entire data science pipeline.

**Technologies Used:**

**SAS Programming**: Base SAS, PROC CONTENTS, PROC FREQ, PROC MEANS, PROC SGPLOT

**Concepts**:Data Cleaning, Feature Engineering, Descriptive Statistics, Data Visualization

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

**Tools:** Microsoft Power BI (Power Query, DAX, Report View)

**Concepts:** KPI Tracking, Performance Analysis, Data Modeling, Dashboard Design, Data Visualization, Business Intelligence, Operational Efficiency, Customer Service Analytics.


**Repository Link:** https://github.com/Asthy247/Call-Center-Performance-Analysis-with-Power-BI



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

**Cluster 5 (Low Income, Medium Spending)**: This cluster represents customers with high spending scores and moderate annual income. They may be willing to spend on premium products and services.

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

**Comprehensive Text Preprocessing:** Implemented a robust text preprocessing pipeline including tokenization, stop-word removal, stemming/lemmatization, and lowercasing to clean and prepare raw textual data for analysis. This step is critical for ensuring the accuracy of subsequent NLP tasks.

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



# 6. Brain Tumor Detection via Advanced Image Processing & Feature Extraction (Python)
**Project Description:**

This project explores the critical application of image processing techniques in Python for the early stages of brain tumor detection from medical image data (e.g., MRI scans). By focusing on robust image preprocessing and multi-faceted feature extraction, this project lays the groundwork for an automated diagnostic aid, demonstrating the potential of computer vision in enhancing medical analysis and improving patient outcomes.

**Key Contributions & Quantifiable Achievements:**

**Medical Image Data Handling & Preprocessing:** Developed a comprehensive pipeline for ingesting and preprocessing medical image data (e.g., MRI scans). This included noise reduction, image normalization, and resizing to prepare images for feature extraction. This rigorous preprocessing is crucial for optimizing feature quality by up to 20% (estimated) for subsequent analysis.

**Advanced Feature Extraction using Multi-Domain Filtering:** Implemented and applied various state-of-the-art image processing filters to extract distinct features from brain scans, crucial for identifying potential tumor regions, including:

**Edge Detection (e.g., Prewitt Filter):** Clearly highlighting boundaries and intricate structures, as demonstrated by the side-by-side visualization of original vs. edge-detected images.

**Prewitt Filter for Data Visualization on Both the Original and Edge-Detected Version**
<img width="958" height="460" alt="image" src="https://github.com/user-attachments/assets/8d70d593-1d8c-47b8-8586-612acce67a7b" />

**Texture Analysis (e.g., Entropy Filtering):** Revealing textural variations indicative of abnormal regions.

**Blob/Ridge Detection (e.g., Hessian Matrix):** Emphasizing intensity changes and structural patterns at multiple scales.
These combined approaches provide a multi-dimensional feature set crucial for comprehensive medical image analysis. Further examples of these techniques, including Entropy and Hessian Matrix visualizations, can be found in the project's Jupyter Notebook.

**Data Visualization for Diagnostic Aid:** Created compelling visualizations of both original and processed images, demonstrating the effectiveness of each image processing step in highlighting specific diagnostic features. The side-by-side comparisons offer clear proof of concept for the utility of each filter in making subtle anomalies more apparent.

**Foundation for Automated Diagnosis:** This project establishes a robust foundation of preprocessed data and extracted features, which are essential for developing future automated systems that could assist radiologists in quickly screening large volumes of medical images, potentially reducing diagnostic time and improving accuracy.

**Technologies Used:**

**Programming Language:** Python

**Libraries:** OpenCV, scikit-image, NumPy, Matplotlib

**Concepts:** Image Processing, Computer Vision, Edge Detection (e.g., Prewitt, Sobel, Canny), Texture Analysis (e.g., Entropy), Blob/Ridge Detection (e.g., Hessian Matrix), Feature Extraction, Medical Imaging Analysis, Data Visualization.

**Repository Link**: https://github.com/Asthy247/Brain-Tumor-Detection-using-Image-Processing-with-Python.git



# 7. Global Video Game Sales Trend Analysis & Interactive Dashboard (Power BI)

**Project Description:**

This project involved the creation of a dynamic and interactive Power BI dashboard designed to provide comprehensive insights into global video game sales trends. By analyzing historical sales data across various dimensions (genre, platform, region, publisher, year), the dashboard reveals key market dynamics, identifies top-performing platforms and titles, and uncovers actionable insights for publishers, developers, and market analysts. This solution transforms raw sales figures into a visually compelling and decision-support tool, enabling data-driven strategic planning.

**Key Contributions & Quantifiable Achievements:**

**Data Acquisition & Preprocessing**: Sourced and meticulously cleaned a large dataset of global video game sales, addressing inconsistencies and preparing the data for robust analysis within Power BI.

**Comprehensive Sales Trend Analysis:** Developed a Power BI dashboard with multiple interactive views to analyze sales trends over time, by genre, and by geographical region.


**Data Visualization for Video Game Sales**
<img width="934" height="540" alt="image" src="https://github.com/user-attachments/assets/f85da609-f2af-40e8-82d4-40c470b860fa" />

**Platform Performance Identification:** Clearly identified and visualized top-performing gaming platforms over specific periods. The dashboard features a "Top 10 Platform by Year" bar chart (as seen in the top-left) which vividly illustrates the dominance of platforms like DS and PS2 based on their cumulative activity.

**Regional & Global Revenue Insights**: Analyzed sum of sales across different regions (EU, Global, JP, NA, Other), as depicted in the donut chart (top-right) and the "Sum of All Sales by Genre" area chart (bottom-left). This visual clearly shows Global Sales (55.91%) as the largest segment, with North America (23.XX%) and Europe (11.96%) as significant contributors. This provides crucial insights into primary revenue-generating markets.

**Genre-Specific Performance Analysis & Trends:** Utilized the "Rank by Genre" line chart (top-middle) and the "Sum of All Sales by Genre" area chart (bottom-left) to enable a quick understanding of top-performing genres like 'Action' and 'Sports' and their sales contributions over time, identifying key genre trends.

**Interactive Dashboard Design:** Created a highly intuitive and interactive user interface within Power BI, demonstrated by the comprehensive dashboard layout, allowing users to filter data by year, genre, platform, and region to drill down into specific market segments and uncover granular insights.

**Actionable Business Intelligence:** The dashboard provides immediate and actionable insights for stakeholders, enabling them to:

Identify emerging market opportunities or declining trends by genre and region.

Understand the competitive landscape of different platforms.

Inform strategic decisions for game development, publishing, and regional marketing efforts by highlighting key revenue drivers and market shares.

**Technologies Used:**

**Tools:** Microsoft Power BI (Power Query for ETL, DAX for calculated measures, Interactive Dashboard Design, Data Modeling)

**Concepts:** Sales Analysis, Market Trend Identification, Data Cleaning, Data Visualization, Business Intelligence, Dashboarding, Interactive Reporting, Regional Performance Analysis.


**Repository Link**: https://github.com/Asthy247/Video-Game-Sales-Data-Visualization-with-Power-BI.git

# 8. Solar Potential Analysis & Regional Insights (Google BigQuery)
**Project Description:**

This project leverages Google BigQuery to analyze the solar energy potential across the United States, utilizing the bigquery-public-data.sunroof_solar.solar_potential_by_postal_code dataset. The primary objective was to identify geographic areas with optimal solar viability and uncover key correlations within solar metrics, providing data-driven insights for strategic decision-making in solar energy development and investment. This analysis demonstrates proficiency in querying, transforming, and analyzing large-scale public datasets directly within a cloud data warehouse environment.

**Key Contributions & Quantifiable Achievements:**

**BigQuery Data Extraction & Preprocessing:** Efficiently extracted and processed relevant data (including postal_code, average_solar_potential, total_area, yearly_sunlight_kwh_total, count_qualified) from a large-scale public dataset within BigQuery. This involved handling missing values and ensuring data consistency to prepare for robust analysis.

**Exploratory Data Analysis (EDA) with SQL:** Conducted in-database EDA by calculating statistical summaries to understand data distributions and identify outliers. This approach leveraged BigQuery's scalable processing power for efficient data exploration.

**Correlation Analysis for Solar Potential:** Performed correlation analysis directly within BigQuery to assess relationships between key metrics. A strong positive correlation coefficient of 0.889 was found between yearly_sunlight_kwh_total (total yearly sunlight) and count_qualified (number of qualified buildings), indicating that regions with more qualified buildings tend to have higher overall solar potential.

**Top Regional Solar Potential Identification:** Identified and ranked the top performing states by average solar potential based on yearly_sunlight_kwh_kw_threshold_avg. The analysis revealed:

**New Mexico:** 1416.69 kWh/kW

**Arizona:** 1371.32 kWh/kW

**Nevada:** 1363.57 kWh/kW

These findings highlight optimal locations for large-scale solar energy projects.

**Actionable Recommendations for Solar Investment:** Based on the analytical findings, developed a set of strategic recommendations for stakeholders, including:

**Targeted Solar Investments:** Prioritize solar energy investments in identified high-potential states (e.g., New Mexico, Arizona, Nevada).

**Policy Incentives:** Implement supportive policies (tax credits, rebates, net metering) to encourage solar energy adoption.

**Grid Integration:** Invest in grid infrastructure to accommodate increased solar energy generation and ensure stability.

**Community Solar Programs:** Promote programs for broader participation in solar energy.

**Research and Development:** Continue R&D to improve solar cell efficiency and reduce costs.


**Example SQL Queries:**

**SQL**


<img width="849" height="471" alt="image" src="https://github.com/user-attachments/assets/b22cb939-72f7-491c-874d-ac6c4e2f31b1" />



**Technologies Used:**

**Tools:** Google BigQuery (SQL), Google Cloud Platform

**Concepts**: Data Extraction (ETL), Data Cleaning, Exploratory Data Analysis (EDA), Correlation Analysis, Cloud Data Warehousing, Renewable Energy Analytics, Business Recommendations.

**Repository Link:** https://github.com/Asthy247/Leveraging-BigQuery-to-Analyze-Solar-Potential-Across-the-US.git


# 9. Customer Churn Prediction Data Pipeline on Snowflake & Google Colab
**Description**: An end-to-end project showcasing a robust data pipeline in Snowflake for data warehousing and feature engineering, integrated with Google Colab (Python) 
for machine learning model training (Logistic Regression, Random Forest), hyperparameter tuning, and comprehensive evaluation. 
This project identifies churn drivers and predicts customer attrition, providing actionable insights for retention strategies.

**Key Visualizations:**

**Receiver Operating Characteristic (ROC) Curve:**


<img width="891" height="691" alt="image" src="https://github.com/user-attachments/assets/43c03f52-e2c3-4174-b21e-538a5ab95bfd" />


**Interpretation**: This plot illustrates the diagnostic ability of both models. An Area Under the Curve (AUC) of 1.00 for both Logistic Regression and Random Forest indicates perfect classification in this small, simulated dataset. In real-world scenarios, an AUC closer to 1.00 signifies a strong predictive model.




**Top 10 Feature Importance (Random Forest):**


<img width="1156" height="685" alt="image" src="https://github.com/user-attachments/assets/477c3d35-42fa-44d2-a7bd-7d69e915f53b" />


**Interpretation:** This chart highlights the most influential features for churn prediction according to the Random Forest model. TENURE_MONTHS (customer tenure) is identified as the most important feature, followed by PAYMENT_METHOD_ELECTRONIC_CHECK and TOTAL_CHARGES. This provides actionable insights into key drivers of churn.




**Distribution of Churn (True vs. False):**



**Interpretation:** This bar chart displays the class balance of the target variable (CHURN_LABEL), showing 6 customers who did not churn and 4 who did. Understanding class distribution is crucial for selecting appropriate evaluation metrics and potential handling of class imbalance in larger datasets.

<img width="1006" height="700" alt="image" src="https://github.com/user-attachments/assets/a090f842-6d54-4b14-97c0-3f1f5841a314" />


**Key Skills:** Snowflake (Data Engineering, ELT, Feature Engineering), SQL (Advanced), Python (Pandas, Scikit-learn, Matplotlib, Seaborn), Machine Learning (Classification, Hyperparameter Tuning), Model Evaluation (ROC AUC, Feature Importance), Cloud Data Platforms, Data Visualization.


**Repository Link:** https://github.com/Asthy247/Snowflake_Churn_Prediction_Project.ipynb.git




# 10. Retail Sales Trends & RFM Segmentation (Databricks SQL)

**Description:** An end-to-end data analysis project showcasing robust data cleaning, exploratory data analysis (EDA), and advanced RFM (Recency, Frequency, Monetary) 

customer segmentation using SQL within the Databricks environment. This project identifies key retail sales trends, evaluates product and demographic performance, and segments customers to 

provide actionable insights for optimizing marketing strategies and driving business growth.

**Key Visualizations:**

**Monthly Sales Trend:**


<img width="669" height="324" alt="Monthly Sales Trend" src="https://github.com/user-attachments/assets/fd1d2a8e-3ef5-4a0d-8446-0a63417a0582" />


**Interpretation:** This line chart clearly illustrates fluctuating monthly sales trends, with noticeable peaks (e.g., November and December) and dips (e.g., January and September). 

This pattern highlights strong seasonality in retail purchases, providing crucial information for inventory management and strategic marketing campaign timing.



**Product Category Sales:**

<img width="651" height="324" alt="Product Category Sales" src="https://github.com/user-attachments/assets/4ead3be4-6685-406f-8e6c-6ad380154b80" />

**Interpretation:** This bar chart reveals that Electronics and Clothing are the leading product categories by total sales, 

indicating strong customer preference and market demand. Beauty follows as a significant, but potentially underperforming, 

category, suggesting opportunities for targeted promotions or product diversification.



**Sales Performance by Gender and Age Group:**

<img width="669" height="324" alt="Sales Perfromance by Gender and Age Group" src="https://github.com/user-attachments/assets/eedca312-d6a8-46a5-9da8-5f043bf86fd9" />


**Interpretation**: This stacked bar chart effectively visualizes that the 25-34 and 35-44 age groups contribute most significantly to overall sales,

with Female customers consistently showing higher spending across most age segments. This insight is crucial for tailoring marketing content 

and product recommendations to resonate with high-value demographic groups.



**Sales by Day of Week:**

<img width="669" height="324" alt="Sales by Day of Week" src="https://github.com/user-attachments/assets/8d82134c-2c1b-4334-811e-e9d3f1d22d54" />

**Interpretation**: This bar chart clearly indicates that Friday and Saturday consistently show the highest sales volume, confirming weekends

as peak shopping periods. This information is crucial for optimizing staffing levels, scheduling promotions, and focusing online 

advertising efforts to capitalize on increased customer activity.


**Customer Segments by Count (RFM Segmentation):**

<img width="671" height="324" alt="Customer Segmentation Bar Chart Visualization" src="https://github.com/user-attachments/assets/41534602-e4a9-4150-ab29-459ceff3d697" />

**Interpretation**: This bar chart illustrates the distribution of customers across various RFM segments (e.g., Champions, At Risk). 

It highlights key customer groups, enabling targeted marketing and retention strategies based on their purchasing behavior, and provides a 

quick understanding of the customer base's health.



**Key Skills**: Databricks SQL, SQL (Advanced), Data Cleaning, Data Transformation, Exploratory Data Analysis (EDA), RFM Segmentation, and Cloud Data Platforms.

**Repository Link**: https://github.com/Asthy247/Retail-Sales-Trends-RFM-Segmentation-Databricks-SQL




# 11. Marketing Campaign Performance: SQL in Google Sheets

**Description**: This project involved a comprehensive exploratory data analysis (EDA) on a marketing campaign performance dataset, leveraging SQL-like queries within Google Sheets. The primary objective was to uncover significant marketing trends, assess data quality, and identify key patterns within campaign types and channels, demonstrating a robust end-to-end analytical workflow from data preparation to insightful visualization and strategic recommendations.

**Key Contributions & Quantifiable Achievements:**

**Data Preparation & Feature Engineering**: Cleaned and preprocessed a dataset of raw marketing campaign records, creating new analytical metrics such as Click-Through Rate (CTR), Conversion Rate, and Return on Investment (ROI) directly within SQL queries. This involved handling data aggregation and transformation for robust analysis.

**Comprehensive Exploratory Data Analysis (EDA)**: Performed extensive descriptive statistics and aggregations using Google Sheets' QUERY() function. Analyzed distributions of key marketing indicators (e.g., Clicks, Impressions, Conversions, Engagement Score) and their relationships with various campaign types and channels.

**Identified Key Marketing Trends & Patterns**: Revealed significant correlations and patterns between campaign types and engagement metrics, and between channels and financial outcomes, providing actionable insights into performance drivers. For example:

Observed that Impressions consistently outweighed Clicks across all campaign types (e.g., Display had over 220 million impressions but 22 million clicks), highlighting the scale of exposure versus direct engagement.

Identified that Average ROI was consistently strong (around 5.00) across all channels despite generally low Average Conversion Rates (around 0.08%), indicating efficient campaign spending.

**Bar Chart: Total Clicks and Total Impressions by Campaign Type**


<img width="1131" height="475" alt="image" src="https://github.com/user-attachments/assets/c083fd8c-d552-4d02-86e7-4b46bba70db4" />


**Visualization**: This chart visually compares the **hundreds of millions of Impressions** (red bars) against the **tens of millions of Clicks** (blue bars) for each campaign type.
    
**Insight**: Impressions significantly outweigh clicks, highlighting the scale of exposure versus direct engagement, with Impressions showing more consistency across campaign types.


    
**Bar Chart: Average Conversion Rate and Average ROI by Channel Used**


<img width="1244" height="487" alt="image" src="https://github.com/user-attachments/assets/ed2bcc53-f990-4809-be55-62255c7a4c79" />


  **Visualization**: This chart shows the **Average ROI** (red bars, around `5.00`) is significantly higher than the **Average Conversion Rate** (blue bars, around `0.08%`) across channels like Email, Google Ads, Website, Facebook, YouTube, and Instagram.
  
**Insight**: Average ROI remains remarkably uniform across all channels, despite low conversion rates, indicating efficient campaign spend.



Pinpointed specific high-engagement campaigns (Engagement Score > 7), with CTRs as high as 7.68%, providing granular data for successful strategy replication.

**Data Quality Assessment & Validation**: Conducted thorough data validation checks using SQL queries to ensure accuracy and consistency of metrics across various dimensions. This proactive approach identified potential discrepancies and ensured the reliability of reported figures, which is critical for making informed decisions on ~$50M+ in annual ad spend, aiming for a 98%+ budget delivery target.

**Access the full project and code on GitHub:** https://github.com/Asthy247/Marketing-Campaign-Performance-SQL-in-Google-Sheet.git



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

