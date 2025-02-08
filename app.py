import streamlit as st
import joblib
import pandas as pd
import numpy as np 

pipeline = joblib.load("NB_pipeline.pkl")
def Prediction(Gender, Married, Dependents, Education,Self_Employed , ApplicantIncome,CoapplicantIncome ,LoanAmount,Loan_Amount_Term ,Credit_History ,Property_Area  ):
            df = pd.DataFrame(columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
            df.at[0, 'Gender'] = Gender
            df.at[0, 'Married'] = Married
            df.at[0, 'Dependents'] = Dependents
            df.at[0, 'Education'] =Education
            df.at[0, 'Self_Employed'] = Self_Employed
            df.at[0, 'ApplicantIncome'] = ApplicantIncome
            df.at[0, 'CoapplicantIncome'] = CoapplicantIncome
            df.at[0, 'LoanAmount'] = LoanAmount
            df.at[0, 'Loan_Amount_Term'] = Loan_Amount_Term
            df.at[0, 'Credit_History'] = Credit_History
            df.at[0, 'Property_Area'] = Property_Area
           

            result = pipeline.predict(df)[0]
            return result
def Main(): 
    Gender = st.selectbox('please select your gender ', ['Male', 'Female'])
    Married = st.selectbox('are you married ', ['Yes', 'No'])
    Dependents = st.selectbox('How many dependents do you have?', [0.0, 1.0, 2.0, 3.0])
    Education = st.selectbox('What is your highest level of education?', ['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox('Are you self employed?',['Yes', 'No'])
    ApplicantIncome = st.number_input('What is your total income?', min_value=0,value=30000,step=1000)
    CoapplicantIncome = st.number_input('What is the coapplicant income?',min_value=0,value=0,step=100) 
    LoanAmount = st.number_input("What is the loan amount you are requesting?", min_value=0, value=100000,step=1000)
    loan_terms = [360, 180, 480, 300, 84, 120, 240, 60, 36, 12]
    Loan_Amount_Term = st.selectbox('How long would you like to repay the loan? Select the term in months's,  options=loan_terms)
    Credit_History = st.selectbox("Do you have a positive credit history?", [0, 1])
    Property_Area = st.selectbox('what is your property area ?',['Urban', 'Rural', 'Semiurban']) 

    if st.button('Predict'):
            result = Prediction(Gender,Married,Dependents,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,Credit_History,Property_Area )
            st.write('### Prediction Result:')
            st.write(f'The predicted result is: {round(np.exp(result), 2)}')
    
    
    
Main()
