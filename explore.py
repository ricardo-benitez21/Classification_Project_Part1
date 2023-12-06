import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def countplot_churn(train):
    '''
    Function that displays a countplot of churned and not churned customers
    '''
    sns.countplot(data=train, x='churn') 


def countplot_senior_citizen(train):
    '''
    Function that displays a countplot churned and not churned senior and non senior citizens
    '''
    sns.countplot(x = 'senior_citizen', hue = 'churn', data = train)

def chi2_senior_citizen(train):
    '''
    Performs a chi-squared test for independence between senior_ citizen and churn.
    '''
    observed = pd.crosstab(train.churn, train.senior_citizen)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between being a senior citizen or not and a customer churning')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between being a senior citizen or not and a customer churning')


def countplot_contract_type(train):
    '''
    FUnction that displays a countplot between contract type and churn
    '''
    sns.countplot(x="contract_type", hue="churn", data=train, palette = 'inferno')

def chi2_contract_type(train):
    '''
    Performs a chi-squared test for independence between contract type and churn.
    '''
    observed = pd.crosstab(train.contract_type, train.churn)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between contract type and churn')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between contract type and churn')

def boxplot_monthly_charges(train):
    sns.boxplot(data=train, y='monthly_charges', x='churn', palette = 'BuPu')
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
    sns.countplot(data=train, x='dependents', hue='churn', palette = 'prism')
    plt.title('Does having dependents affect churn')
    plt.show()

def chi2_dependents(train):
    observed = pd.crosstab(train.contract_type, train.churn)
    alpha = 0.05
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    print(f'p-value: {p}')
    if p < alpha:
        print('We can reject our null hypothesis and say that there is a relationship between dependents and churn')
    else:
        print('We fail to reject our null hypothesis and say that there is NO relationship between dependents and churn')