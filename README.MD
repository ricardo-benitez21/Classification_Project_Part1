# Telco Customers Churning

# Project Description

There can be many factors that lead to a customer churning. One factor I will focusing on is "Do non senior citizens churn more than senior citizens?"

# Project Goals

* Discover drivers of customers churning
* Use these drivers to develop a machine learning model to see if features resutls in churning
* Use the findings to further understand the customers and develop a buisness plan pertaining to senior citzens or non senior citzens.

# Initial Thoughts

My hypothesis is that non-senior citizens churn more than senior citizens. Usually senior citzens (62 or older) tend to just stay with one company since they might not have the means of exploring other companies. Non senior citizens, such as myself, are usually trying to find the best bang for our buck when it comes to subscription services.

# The Plan

* Acquire data from the telco_churn csv file

* Prepare the data
    * Will be using a prepare.py file to do this

* Explore the data to see if there are other drivers that lead to non seniors churning
    * Ask and answer the following questions
        * Do dependents factor into chrun?
        * Does having a longer tenure there result in staying or churning?
        * Do  non senior citizens have higher monthly charges than senior citizens?

* Come up with a model to predict if non senior citizens churn more than senior citizens
    * Use the drivers identified in explore to build predictive models of differnt types
    * Use the train and validate data to evaulate models
    * Choose the best model based on highest accuracy 
    * Evauluate the best model on test data

* Draw conclusions

# Data Dictionary

| Feature | Definition |
|:--------|:-----------|
|customer_id| id used to identify customer|
|gender| gender of the customer (Male or Female)|
|senior_citizen| determines if customer is a senior citizen or not ( 0 = False, 1 = True)|
|dependents| customer either has dependents or no dependents|
|tenure| how long a customer has been with the company (in months)|
|phone_service| does customer have phone service or not|
|multiple_lines| does customer have multiple lines or not|
|online_security| does customer have online security or not|
|online_backup| does customer have online backup or not|
|device_protection| does customer have device protection or not|
|tech_support| does customer have tech support or not|
|streaming_tv| does customer stream tv or not|
|streaming_movies| does customer stream movies or not|
|paperless_billing| does customer use paperless billing or not|
|monthly_charges| how much a customer gets charged monthly|
|total_charges| the total amount of charges a customer has totaled since they joined the company|
|churn| did the customer churn or not|
|contract_type| the type of contract a customer has|
|internet_service_type| the type of internet a customer has|
|payment_type| how a customer pays

# Steps to Reproduce
1) Clone this repo.
2) Acquire the telco_churn csv file
3) Put the data in the file containing the cloned repo.
4) Run notebook