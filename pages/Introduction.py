import streamlit as st
import pandas as pd
st.title("User Notice")
st.markdown("### Before the evaluation, you have to read the followings seriously.")
st.markdown("""
- #### We have to get some of your information. Don't worry about the privacy, we just use it for training.
- #### In this project, we use four different models to evaluate whether the lender has the ability to repay the loan. The chart below shows the accuracy of each model.
""")
df = pd.DataFrame({
    "Model": ["XGBClassifier", "SVC", "RandomForestClassifier", "GradientBoostingClassifier"],
    "Accuracy": [0.903, 0.878, 0.902, 0.897]
})
st.dataframe(df, width=500)
st.markdown("""### We have to get some information listed below, we also give the introduction of each feature:
- 1 - age
- 2 - job 
- 3 - marital
- 4 - education 
- 5 - default: has credit in default? 
- 6 - housing: has housing loan?
- 7 - loan: has personal loan? 
- 8 - contact: contact communication type
- 9 - month: last contact month of year 
- 10 - day_of_week: last contact day of the week 
- 11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target.
- 12 - campaign: number of contacts performed during this campaign and for this client
- 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- 14 - previous: number of contacts performed before this campaign and for this client
- 15 - poutcome: outcome of the previous marketing campaign 
- 16 - emp.var.rate: employment variation rate - quarterly indicator 
- 17 - cons.price.idx: consumer price index - monthly indicator 
- 18 - cons.conf.idx: consumer confidence index - monthly indicator 
- 19 - euribor3m: euribor 3 month rate - daily indicator
- 20 - nr.employed: number of employees - quarterly indicator 
""")
