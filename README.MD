# Why are Telco Customers Churning?

# Project Description

This project will try to find out why some Telco customers are churning.

# Project Goals

* Discover drivers of customers churning
* Use these drivers to develop a machine learning model to see if features chosen result in churning
* Use the findings to further understand the customers and develop a plan on how to keep customers from churning

# Initial Thoughts

My hypothesis is that the features below affect whether or not a customer will churn or not :

    * Senior Citizen - Senior citizens (62 and older) tend to stay with a provider since they might not know how to switch to other providers. Non senior citizens, like myself, are always looking for service providers that will give us the best bang for our buck. This can lead to non senior citizens churning.

    * Tenure - The amount of time a customer has been with a provider is important. A short tenure amount can indicate churning.
    
    * Monthly Charges - If a customer sees that their monthly charges are high, then that can lead to churn.

# The Plan

* Acquire data from MySql 

* Prepare the data
    * Will be using a prepare.py file to do this
    * Split the data into train (60%), validate (20%), and test (20%) dataframes

* Explore the data to see if there are other drivers that lead to churning
    

* Modeling
    * Use the drivers identified in explore to build predictive models of different types
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
|phone_service| customer has phone service or not|
|multiple_lines| customer has multiple lines or not|
|online_security| customer has online security or not|
|online_backup| customer has online backup or not|
|device_protection| customer has device protection or not|
|tech_support| customer has tech support or not|
|streaming_tv|  customer streams tv or not|
|streaming_movies| customer streams movies or not|
|paperless_billing| customer uses paperless billing or not|
|monthly_charges| how much a customer gets charged monthly|
|total_charges| the total amount of charges a customer has totaled since they joined the company|
|churn| did the customer churn or not|
|contract_type| the type of contract a customer has|
|internet_service_type| the type of internet a customer has|
|payment_type| how a customer pays

# Steps to Reproduce
1) Clone this repo.
2) Create an env.py file that follows the following format : 
host = 'data.codeup.com'
username = 'sample_username'
password = 'sample_password'

def get_db_url(database_name, host_name=host, password=password, username=username):
    return f'mysql+pymysql://{username}:{password}@{host_name}/{database_name}'

3) Save the env.py file into the repo
4) Run the notebook
