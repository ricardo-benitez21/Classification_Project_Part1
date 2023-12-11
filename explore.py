import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def countplot_churn(train):
    '''
    Function that displays a countplot of churned and not churned customers
    '''
    sns.countplot(data=train, x='churn', palette = 'prism') 


def barplot_senior_citizen(train):
    '''
    Function that displays a countplot churned and not churned senior and non senior citizens
    '''
    sns.barplot(x = 'senior_citizen', y = 'churn', data = train, palette = 'prism')

def chi2_senior_citizen(train):
    '''
    Performs a chi-squared test for independence between senior_ citizen and churn.
    '''
    observed = pd.crosstab(train.churn, train.senior_citizen)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between being a senior citizen and churn')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between being a senior citizen and churn')


def countplot_gender(train):
    '''
    Function that displays a countplot between dependents and churn
    '''
    sns.countplot(x="gender", hue="churn", data=train, palette = 'prism')

def chi2_gender(train):
    '''
    Performs a chi-squared test for independence between contract type and churn.
    '''
    observed = pd.crosstab(train.gender, train.churn)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between being a female and churn')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between being a female and churn')

def boxplot_monthly_charges(train):
    '''
    Function that creates a boxplot between monthly charges and churn
    '''
    sns.boxplot(data=train, y='monthly_charges', x='churn', palette = 'prism')
    plt.title('Do customers that churn pay more per month?')
    plt.show()

def ttest_monthly_charges(train):
    churn = train[train.churn == 'Yes'].monthly_charges
    no_churn = train[train.churn == 'No'].monthly_charges
    t, p = stats.ttest_ind(churn, no_churn)
    print(f'p-value: {p}')
    if p < .05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

def countplot_dependents(train):
    '''
    Function that creates a countplot between dependents and churn
    '''
    sns.countplot(data=train, x='dependents', hue='churn', palette = 'prism')
    plt.title('Does having dependents affect churn')
    plt.show()

def chi2_dependents(train):
    '''
    Performs a chi-squared test for independence between dependents and churn.
    '''
    observed = pd.crosstab(train.contract_type, train.churn)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between dependents and churn')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between dependents and churn')