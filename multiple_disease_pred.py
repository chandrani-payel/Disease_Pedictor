# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 23:20:57 2022

@author: chandrani
"""

import pickle
import joblib
import time
import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from streamlit_option_menu import option_menu

import sqlite3
#import streamlit as st1


import streamlit.components as stc 
import base64 

import openai
import aiohttp
import streamlit_chat
from streamlit_chat import message

from dotenv import load_dotenv
load_dotenv()

import time
timestr = time.strftime("%Y%m%d-%H%M%S")


# loading the saved models


diabetes_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model/diabetes_model (1).sav"), 'rb'))

heart_disease_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model/heart_disease_model (1).sav"),'rb'))
                                  

parkinsons_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model/parkinsons_model (1).sav"),'rb'))
                              
model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_model/random_f.joblib"))
#model = joblib.load(open('C:/Users\sabyasachi\Desktop\Multiple_Disease_Prediction\saved_model\random_f.joblib', 'rb'))

#Welcome text
st.title("Welcome to _Predico_ - The Smart Health Predictor")
# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Covid-19 Prediction',
                           'General Disease Prediction',
                           'Predo-Bot'
                           ],
                          icons=['activity','heart','arrow-up-right-circle-fill','badge-vr-fill','person-fill','robot'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    
    # getting the input data from the user
    name = st.text_input("Enter your Name")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)
    
    #st.download_button(label='Download File',data=Diabetes Prediction.to csv(),mime='text/csv')

# Database for Diabetes
                

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')
    
    name = st.text_input("Enter Your Name")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age:')
        
    with col2:
        sex = st.number_input('Sex: 1 = Male; 0 = Female')
        
    with col3:
        cp = st.number_input('Chest Pain types')
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure')
        
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.number_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise')
        
    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.number_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        

# Database for Heart Disease
    	
                
    

# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):
    
    # page title
    st.title("Parkinson's Disease Prediction using ML")
    
    name = st.text_input("Enter Your Name")
    
    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

# Covid-19 Prediction Page
if (selected == 'Covid-19 Prediction'):
    
    # page title
    st.title('Covid-19 Prediction')
    
    #loading the Covid-19 dataset
    #df1=pd.read_csv("C:/Users\sabyasachi\Desktop\Multiple_Disease_Prediction\covid.csv")
    
    df1=pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "covid.csv"))
    #cleaning the data by dropping unneccessary column and dividing the data as features(x1) & target(y1)
    x1=df1.drop("Infected with Covid19",axis=1)
    x1=np.array(x1)
    y1=pd.DataFrame(df1["Infected with Covid19"])
    y1=np.array(y1)
    #performing train-test split on the data
    x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
    #creating an object for the model for further usage
    model1=RandomForestClassifier()
    #fitting the model with train data (x1_train & y1_train)
    model1.fit(x1_train,y1_train)
    
    #Covid-19 Page
    
    #heading over to the Covid-19 section
    #if rad=="Covid-19":
    #st.header("Know If You Are Affected By Covid-19")
    st.write("All The Values Should Be In Range Mentioned")
        #taking the 4 most important features as input as features -> Dry Cough (drycough), Fever (fever), Sore Throat (sorethroat), Breathing Problem (breathingprob)
        #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
        #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
        
    name = st.text_input("Enter Your Name")
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
        #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]
        
        #prediction part predicts whether the person is affected by Covid-19 or not by the help of features taken as input
        #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction1=="Yes":
            st.warning("You Might Be Affected By Covid-19")
        elif prediction1=="No":
            st.success("You Are Safe")
    
# Database for Covid-19

 

#  General Disease Prediction Page
if (selected == 'General Disease Prediction'):
    
    # page title
    st.title('General Disease Prediction')
    
    # get input data as symtoms from user
   
   
    symptoms_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
    
    # start of streamlit UI
    #st.title("Predico - The Smart health predictor")
    #st.header("Please enter your symptoms 🩺")
    
    symptoms = st.multiselect('Enter your symptoms so that we can get you a primary diagnosis:',[*symptoms_list],key='symptoms')
    
    # 
    prediction_value = [0 for i in range(132)]
    for sym in symptoms:
        index = symptoms_list.index(sym)
        # assigning encoded value to testing frame
        prediction_value[index] = 1
    
    # convert list to Pandas dataframe and transpose it for model evaluation
    query = pd.DataFrame(prediction_value).T
    prediction = model.predict(query)
    
    # evaluation and confirmation
    if st.button("Evaluate"):
        with st.spinner('Predicting output...'):
            time.sleep(1)
            if symptoms:
                st.success("Prediction complete!")
                st.write("The diagnosis we have reached is: ")
                st.error(*prediction)
                st.write("Please consult your nearest health administrator soon, take care! 🏥")
                
            else:
                st.info("Please enter at least one symptom before clicking evaluate!")
                
    
# Database for General Disease



# Chat Bot code
# Predo-Bot
if (selected == 'Predo-Bot'):

    st.header(" _Predo_ is here for your Service ")
    
    openai.api_key = st.secrets["OPENAI_SECRET"]
#function to generate call from api

    def generate_response(prompt):
        completions = openai.Completion.create(
            engine = "text-davinci-003",
            prompt=prompt,
            max_tokens = 1024,
            n=1,
            stop = None,
            temperature = 0.5
        )
        message = completions.choices[0].text
        return message
    
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
        

    if 'past' not in st.session_state:
        st.session_state['past'] = []
    
    def get_text():
        input_text = st.text_input("You: ","Hello, How are You? ", key="input")
        return input_text 
    
    user_input = get_text()
    
    if user_input:
        output = generate_response(user_input)
        #store the output
        
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        
    if st.session_state['generated']:

        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

