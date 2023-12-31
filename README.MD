# Why are Telco Customers Churning?

# Project Description

This project will try to find out why some Telco customers are churning.

# Project Goals

* Discover drivers of customers churning
* Use these drivers to develop a machine learning model to see if features chosen result in churning
* Use the findings to further understand the customers and develop a plan on how to keep customers from churning

# Initial Thoughts

My hypothesis is that the features below affect whether or not a customer will churn or not :

    * Senior Citizen - Senior citizens (62 and older) could churn depending on how easy it is to navigate their subscription.

    * Dependents - If you have no dependents, then maybe you might not need all the services compared to customers who do have dependents. Customers with dependents might need to pay for more services, for example you might need to get tv streaming on top of the services you are already payng for for your kids so they can watch their programs.

    * Monthly Charges - If a customer sees that their monthly charges are high, then that can lead to churn.

    * Gender (Female) - Do females churn more than males? 

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

* Conclusions
    * In conclusion, this project has helped us gain some valuable knowledge in regards to Telco and their customers. We can now come up with business plans to help reduce churn rates in the future and to also grow the relationships we have with current customers

* Recommendations
    * For customers who are have high monthly charges, I would recommened providing targeted retention offers, such as discounts or additional services, to incentivize them to stay.

    * For customers who do not have dependents, maybe creating customized offers or promotions that align with the lifestyle of individuals without dependents will encourage them to stay.

    * For customers who are senior citizens, Providing personalized and patient customer support that caters to the specific needs of senior citizens could help reduce the churn of customers who are senior citizens.

* Next Steps
    * With more time, I could look deeper into monthly charges and check to see what exactly is causing their high charges.

    * Come back to the model and update it with any new findings.



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
