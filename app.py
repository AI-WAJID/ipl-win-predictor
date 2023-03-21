import streamlit as st
import pickle
import pandas as pd

teams = ['Sunrisers Hyderabad',         'Royal Challengers Bangalore',         'Rajasthan Royals',         'Gujarat Lions',         'Delhi Capitals',         'Mumbai Indians',         'Kolkata Knight Riders',         'Chennai Super Kings',         'Kings XI Punjab']

cities = ['Bangalore', 'Chandigarh', 'Mumbai', 'Kolkata', 'Jaipur',          'Chennai', 'Hyderabad', 'Cape Town', 'Port Elizabeth', 'Durban',          'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Cuttack',          'Ahmedabad', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Ranchi',          'Delhi', 'Abu Dhabi', 'Pune', 'Rajkot', 'Kanpur', 'Indore',          'Bengaluru', 'Dubai', 'Sharjah']

pipe = pickle.load(open('pipe.pkl', 'rb'))

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', teams)

with col2:
    bowling_team = st.selectbox('Select the bowling team', teams)

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Overs completed')

with col5:
    wickets = st.number_input('Wickets out')

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team], 'bowling_team':[bowling_team], 'city':[selected_city], 'runs_left':[runs_left],
                  'balls_left':[balls_left], 'wickets':[wickets], 'total_runs_x':[target], 'crr':[crr], 'rrr':[rrr]})

    st.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "_ " + str(round(win*100)) + "%")
    st.header(bowling_team + "_ " + str(round(loss*100)) + "%")

