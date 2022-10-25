import joblib
import streamlit as st
import numpy as np
from pandas.core.frame import DataFrame
import pandas as pd
from pages.library.data_mapping import DATA_MAPPING_SVC, DATA_MAPPING_RANDOM
from pages.library.data_processing import DATA_PROCESSING

st.header("Evaluation System")
st.info("MAKE SURE YOU HAVE READ THE INTRODUCTION!!!")

form = []
age = st.slider('How old are you?', 0, 80, 27)
form.append(age)
job = st.selectbox('What\'s your job?',
                   ('unknown', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed'))
form.append(job)
if job == 'unknown':
    job = np.nan
marital = st.selectbox('What\'s your marital status?',
                       ('unknown', 'divorced', 'married', 'single'))
form.append(marital)
if marital == "unknown":
    marital = np.nan
education = st.selectbox('What\'s your education status?',
                         ('basic 4-year education', 'basic 6-year education', 'basic 9-year education', 'high school', 'illiterate', 'professional course', 'university degree', 'unknown'))
form.append(education)
if education == 'basic 4-year education':
    education = 'basic.4y'
elif education == 'basic 6-year education':
    education = 'basic.6y'
elif education == 'basic 9-year education':
    education = 'basic.9y'
elif education == "high school":
    education = 'high.school'
elif education == 'professional course':
    education = 'professional.course'
elif education == 'university degree':
    education = 'university.degree'
elif education == 'unknown':
    education = np.nan
default = st.selectbox('Do you have redit in default',
                       ('no', 'yes', 'unknown'))
form.append(default)
if default == "unknown":
    default = np.nan
housing = st.selectbox('Do you have housing loan?',
                       ('no', 'yes', 'unknown'))
form.append(housing)
if housing == "unknown":
    housing = np.nan
loan = st.selectbox('Do you have personal loan?',
                    ('no', 'yes', 'unknown'))
form.append(loan)
if loan == "unknown":
    loan = np.nan
contact = st.selectbox('What\'s your contact communication type?',
                       ('cellular', 'telephone'))
form.append(contact)
month = st.selectbox('What\'s your last contact month of year?',
                     ('mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
form.append(month)
day_of_week = st.selectbox('What\'s the last contact day of th week?',
                           ('mon', 'tue', 'wed', 'thu', 'fri'))
form.append(day_of_week)
duration = st.slider('What\'s the last contact durations(in seconds)?', 0, 5000, 0)
form.append(duration)
campaign = st.slider('What\'s the number of contacts performed during this campaign and for this client(If over 100 then choose 100)?', 0, 100, 5)
form.append(campaign)
pdays = st.selectbox('What\'s the number of days that passed by after the client was last contacted from a previous campaign(0 means client was not previously contacted)?',
                     (x for x in range(50)))
form.append(pdays)
if pdays == 0:
    pdays = 999
previous = st.slider("What\'s is the number of contacts performed before this campaign and for this client", 0, 10, 0)
form.append(previous)
poutcome = st.selectbox("What\'s is the outcome of the previous marketing campaign ",
                        ('failure', 'nonexistent', 'success'))
form.append(poutcome)
emp_var_rate = st.slider("What\'s the employment variation rate?", -5.0, 5.0, 0.0)
form.append(emp_var_rate)
cons_price_idx = st.slider("What\'s the consumer price index?", 90.0, 100.0, 0.0)
form.append(cons_price_idx)
cons_conf_idx = st.slider("What\'s the consumer confidence index?", -50.0, 0.0, 0.0)
form.append(cons_conf_idx)
euribor3m = st.slider("What\'s the euribor 3 month rate?", 0.000, 5.500, 1.000)
form.append(euribor3m)
nr_employed = st.slider("What\'s the number of employees of quarterly indicator?", 4900, 5200, 5000)
form.append(nr_employed)
if st.button("Evaluate", help="Make Sure You Fill In All The Form"):
    L = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    key = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
           'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
           'cons.conf.idx', 'euribor3m', 'nr.employed']
    l = DataFrame([form], columns=key)
    st.markdown("### The data of yours")
    st.table(l)
    SVC_model = joblib.load("pages/model/SVC.pkl")
    XGBClassifier = joblib.load('pages/model/XGBClassifier.pkl')
    RandomForestClassifier = joblib.load('pages/model/RandomForestClassifier.pkl')
    GradientBoostingClassifier = joblib.load('pages/model/GradientBoostingClassifier.pkl')
    L_svc = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays,
         previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    svc_result = SVC_model.predict(DATA_MAPPING_SVC(L_svc)).astype(int)[0]
    L_ran = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays,
         previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    random_result = RandomForestClassifier.predict(DATA_MAPPING_RANDOM(L_ran)).astype(int)[0]
    L_XGB = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays,
         previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    XGBC_result = XGBClassifier.predict(DATA_PROCESSING(L_XGB)).astype(int)[0]
    L_Gra = [age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays,
         previous, poutcome, emp_var_rate, cons_price_idx, cons_conf_idx, euribor3m, nr_employed]
    Grad_result = GradientBoostingClassifier.predict(DATA_PROCESSING(L_Gra)).astype(int)[0]
    res = []
    res.append([svc_result, random_result, XGBC_result, Grad_result])
    ans = sum(res[0])
    for i in range(len(res[0])):
        if res[0][i] == 0:
            res[0][i] = 'Not Likely'
        else:
            res[0][i] = 'Likely'
    result = {
        "Model": ["XGBClassifier", "SVC", "RandomForestClassifier", "GradientBoostingClassifier"],
        "Prediction": res[0]
    }
    st.table(result)
    if ans >= 3:
        st.balloons()
        st.success("After evaluation, the user is likely to buy savings products!")
    else:
        st.error("Unfortunately, the user is not likely to buy savings products!")
