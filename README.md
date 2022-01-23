# COVID-19 for the High-Risk

<img width="749" alt="covid-19" src="https://user-images.githubusercontent.com/88418201/150701093-20516919-e2c9-4890-9eee-7a4367970298.png">

----------
### An analysis of COVID-19 mortality in patients with pre-existing medical conditions.

## Project Description 

This project aims to use a machine learning classification model to predict COVID-19 mortality based on a patient's demographics and pre-existing health conditions.

- This dataset is on the individual patent level and includes a patient's basic demographics, binary values for having common underlying health conditions, COVID-19 result status, ICU and intubation status, and date of death (if applicable).
- We have analyzed this dataset through machine learning to predict several patient outcomes (ICU entry, intubation, and death) based on their demographics and underlying health conditions.
- Our target variable for mortality prediction is the `date_died` column,  which provides a date value for patient death or a 9999-99-99 for patient survival, and is used to create a new `survived` column of binary values to use in our classification model.
- The `ICU` and `Intubed` columns are similarly set as binary values (1-yes or 2-no) based on whether a patient experienced ICU entry or intubation. These target variables are used individually within the classification model, along with the same features as the the mortality analysis, to predict a patient's experience. 
- In addition to predicting patient outcome, we also look at feature importance within the machine learning model as a way to see which underlying conditions are most likely to contribute to patient mortality.


**[Click Here](https://docs.google.com/presentation/d/13k2VGWm_J2tI8rKIRiugHNP4i3qLytitx4pIWJaisyA/edit?usp=sharing) for the Google Slides presentation on this topic.**
**See Google Slides for storyboard outline of Tableau Dashboard*

## Project Team Members

The following are the members contributing to this project:

	- Alena Swann
	- Anshu Malini
	- Michael Williams
	- Sajini Thiagaraj
	- Sheetal Tondwalkar
	- Shreya Srivastava

## Team Communication

	- Zoom call every alternate day to discuss the progress 
	- Group slack channel for all discussions

## Tools

    * Creating database
        * PostgreSQL
        * Amazon Web Services(AWS)
        
    * Connecting to database
        * Psycopg2
        
    * Analyzing Data
        * Jupyter Notebook
        * Pandas
        
    * Dashboard and Presentation
	    * Tableau Public
    	* Google Slides
    

## Data selection and questions we hope to answer with the data

The deadly disease known as COVID-19, caused by the infectious SARS-CoV-2 virus, has been a pandemic sweeping our world since late 2019 and has been at the forefront of world news, research, and crisis management.   The early days of the pandemic had many unknowns, but one trend was beginning to form: COVID-19 symptoms were more severe and mortality was higher in patients with certain underlying health conditions, deeming them 'high-risk'. 

Since then, significant data on COVID-19 patients has been collected and compiled to help better understand the virus and its severity in patients given certain conditions. Machine learning models can be applied to find correlations between COVID-19 mortality and pre-existing health conditions, providing more insight into who is at high risk of a severe case of COVID-19. This insight can be used within hospital resource management and triage prioritization of high-risk patients. 

Mexico's [Open Data General Directorate of Epidemiology](https://www.gob.mx/salud/documentos/datos-abiertos-152127 "Open Data General Directorate of Epidemiology") COVID-19 database was selected for this predictive study as it provides clear, patient-level, categorical data that is ideal for machine learning. Given the large (and daily-growing) size of this database, we use a subset of the data (1/1/2020-5/31/2020) that has been partially cleaned and obtained from [Kaggle](https://www.kaggle.com/tanmoyx/covid19-patient-precondition-dataset).


## Database

The dataset will be loaded into a AWS RDS database instance by building a connection to PostgreSQL AWS server  and then connected to Jupyter Notebook for machine learning model manipulation. The initial master data table schema is as follows:


#### Column Descriptions
	-	ID	Case identifier number.
	-	SEX	Identifies the sex of the patient.
	-	PATIENT_TYPE Identifies the type of care received by the patient in the unit. It is called an outpatient if you returned home or it is called an inpatient if you were admitted to hospital.
	-	ENTRY_DATE	Identifies the date of the patient's admission to the care unit.
	-	DATE_SYMPTOMS	Identifies the date on which the patient's symptoms began.
	-	DATE_DIED	Identifies the date the patient died.
	-	INTUBED	Identifies if the patient required intubation.
	-	PNEUMONIA	Identifies if the patient was diagnosed with pneumonia.
	-	AGE	Identifies the age of the patient.
	-	PREGNANCY	Identifies if the patient is pregnant.
	-	DIABETES	Identifies if the patient has a diagnosis of diabetes.
	-	COPD	Identifies if the patient has a diagnosis of COPD.
	-	ASTHMA	Identifies if the patient has a diagnosis of asthma.
	-	INMSUPR	Identifies if the patient has immunosuppression.
	-	HYPERTENSION	Identifies if the patient has a diagnosis of hypertension.
	-	OTHER_DISEASE	Identifies if the patient has a diagnosis of other diseases.
	-	CARDIOVASCULAR	Identifies if the patient has a diagnosis of cardiovascular disease.
	-	OBESITY	Identifies if the patient is diagnosed with obesity.
	-	RENAL_CHRONIC	Identifies if the patient has a diagnosis of chronic kidney failure.
	-	TOBACCO	Identifies if the patient has a smoking habit.
	-	CONTACT_OTHER_COVID	Identifies if the patient had contact with any other case diagnosed with SARS CoV-2
	-	COVID_RES	Identifies the result of the analysis of the sample reported by the laboratory of the National Network of Epidemiological Surveillance Laboratories.
	-	ICU	Identifies if the patient required to enter an Intensive Care Unit.


## Exploratory Data Analysis

-----
### Data Preprocessing

1. We have used Python pandas to load the raw data  into the database and then export it into a dataframe for dara cleansing so that we can analyze and make better predictions.
2. Consolidated the data from various sources by removing duplicates to maintain accuracy and to avoid misleading statistics.
3. We have excluded the covid patients records from our analysis whose results were pending.
4. Formatted the date columns (entry_date,date_symptoms,date_died) into a standard mm-dd-yyyy date format.
5. Converted the date_died column into categorical data by populating it into a new `survived` column  for better predictions during the Machine Learning phase.

### Data Loading

We have chosen Amazon Web Services (AWS) relational database system to store data for our project. We are using PostgreSQL; psycopg2 as the adapter to connect the database with our Python code and using SQLAlchemy which is a Python SQL toolkit to facilitate the communication between pandas and the database.

Our database is named `covid19_data_analysis` that stores the static data in four different tables, for our use during the course of the project.

Below is the entity relation diagrams, showing the relationship between the four tables and their columns:

![ERD_covid19_data_analysis](https://user-images.githubusercontent.com/88418201/150700027-e0f1169b-eea0-4194-82f7-11cfdd5f1f9d.png)


## Machine Learning Model

- SVM machine learning is best used when the output data needs to be classifed two categories. For this dataset, did the covid patient die or not die would be our classification.
- To make the best prediction for our dataset, we will try many different classification algorithms for our problem.

#### List of tasks to be performed to achieve our goal:

- Read the data file
- Define the Features and Target Variable
- Split the Data into Training and Testing sets
- Train our Model for different Classification Algorithms namely Decision Tree, SVM Classifier, Random Forest Classifier.
- Select the best Algorithm.





    
