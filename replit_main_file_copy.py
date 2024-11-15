import streamlit as st
import pandas as pd
import pickle
import numpy as np
# numpy is the numerical python library
import os
from openai import OpenAI
import utils as ut

# initialize openAI client with groq endpoint
client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))
# openAI provides a standard api that other api providers have adopted
# meaning we can use alternative
# api's by changing the base_url and api_key
# LPU => Language Processing Unit (Groq has fastest text generation due to them creating their own LPU)


def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_clf.pkl')

xgboost_SMOTE_model = load_model('xgboost-SMOTE.pkl')

xgboost_featureEngineered_model = load_model('xgbost-featureEngineered.pkl')


# define a function to prepare the data to be formatted as a dictionary
def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):

    # data trained with one-hot encoding
    # now dict formatting matches trained model data
    # 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    #  'IsActiveMember', 'EstimatedSalary', 'Geography_France',
    #  'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male',
    #  'CLV', 'TenureAgeRatio', 'AgeGroup_MiddleAge', 'AgeGroup_Senior',
    #  'AgeGroup_Elderly'
    input_dict = {
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': int(has_credit_card),
        'IsActiveMember': int(is_active_member),
        'EstimatedSalary': estimated_salary,
        'Geography_France': 1 if location == "France" else 0,
        'Geography_Germany': 1 if location == "Germany" else 0,
        'Geography_Spain': 1 if location == "Spain" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'CLV': balance * estimated_salary / 100000,
        'TenureAgeRatio': tenure / age,
        'AgeGroup_MiddleAge': 1 if age > 30 and age <= 45 else 0,
        'AgeGroup_Senior': 1 if age > 45 and age <= 65 else 0,
        'AgeGroup_Elderly': 1 if age > 65 else 0,
        # 'AgeGroup_Young': 1 if age <= 30 else 0,
    }

    input_df = pd.DataFrame([input_dict])
    return input_df, input_dict


# explain model predictions
def explain_predictions(probability, input_dict, surname):

    # TODO: Improve prompt
    prompt = f"""You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models. 

  Your machine learning model has predicted that a customer named {surname} has a {round(probability*100, 1)}% chance of churning. This prediction is based on the information provided below.

  Here is the customers information:
  {input_dict}

  Here are the machine learning model's top 10 most important features for predicting churn:

             Feature | Importance
  ----------------------------------------
       NumOfProducts | 0.323888
      IsActiveMember | 0.164146
                 Age | 0.109550
   Geography_Germany | 0.091373
             Balance | 0.052786
    Geography_France | 0.046463
     Geography_Spain | 0.036855
         CreditScore | 0.035005
     EstimatedSalary | 0.032655
           HasCrCard | 0.031940
              Tenure | 0.030054
         Gender_Male | 0.000000

  {pd.set_option('display.max_columns', None)}

  Here are summary statistics for the churned customers:
  {df[df['Exited'] == 1].describe()}

  Here are summary statistics for the non-churned customers:
  {df[df['Exited'] == 0].describe()}

  - If the customer has over a 40% risk of churning, generate a 3 sentence explanation of why they are at risk of churning.
  - If the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk of churning
  - Your explanation should be based on the customer's information, the summary statistics or churned and non-churned customers, and the most important features provided.
  - Your explanation should not list out the customer's information, but instead focus on the most important features and why they are important.
- You should explain the model's predictions in a way that is easy for a non-expert to understand.
- Any prediction percent under 30% is considered a low to no risk of churning.
- Any prediction percent over 60% is considered a high risk of churning.
- Don't mention the data directly instead explain it and why it is important. For example: "I predicted that the customer has a low risk of churning because they have multiple products and are older in age."
- You can compare the customer's information to the summary statistics to understand the customer's risk of churning.

  Don't mention the probability of churning, or the machine learning model. Also please to not say anything like "Based on the machine learning model's predictions and the top 10 most important features". Only explain the prediction not the model itself.


  """
    print("explanation PROMPT", prompt)

    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                  messages=[{
                                                      "role": "user",
                                                      "content": prompt
                                                  }])
    return raw_response.choices[0].message.content


# generate email
def generate_email(probability, input_dict, explanation, surname):
    prompt = f"""You are a manager at HS Bank. You are responsible for ensuring customers stay with the bank and are incentivized with various offers.

  You noticed a customer named {surname} has a {round(probability*100, 1)}% chance of churning. This customer's information is:
  {input_dict}

 Here is an explanation as to why the customer might be at risk of churning:
 {explanation}

 Generate an email to send to the customer based on the customer's information to ask them to stay if they are at risk of churning, or offereing them incentives so that they become more loyal to the bank.

 Make sure to list out a set of incentives to stay based on their information, in bullet point format. Don't ever mention the probability of churning, or the machine learning model to the customer.
  
  """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content


# predictions of the probabilities of each model
def make_predictions(input_df, input_dict):

    probabilities = {
        # [0][1] gets the probability of the positive class/ the probability the customer will churn in %
        'XGBoost': xgboost_model.predict_proba(input_df)[0][1],
        'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
        'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
    }
    # avg of all model predictions
    avg_probability = np.mean(list(probabilities.values()))

    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)
        st.write(
            f"The customer has a {avg_probability:.2%} probability of churning."
        )
    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)

    # displays models probabilities on the frontend
    st.markdown("### Model Probabilities")
    for model, prob in probabilities.items():
        st.write(f"{model} {prob}")
    st.write(f"Average Probability: {avg_probability}")

    return avg_probability


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

# create dropdown to select customers
# iterates through customers and creates a string for each customer with their ID and surname
customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

#  when user selects a customer, the customer's data is displayed

if selected_customer_option:
    # get customer id by splitting the string and taking the first part
    selected_customer_id = int(selected_customer_option.split(" - ")[0])

    # get surname by splitting the string and taking the second part
    selected_customer_surname = selected_customer_option.split(" - ")[1]
    # print("Surname:", selected_customer_surname, "Id:", selected_customer_id )

    # filter the dataframe to only include the selected customer
    # iloc is used to select rows by their index and provides a 1-dimensional array
    selected_customer = df.loc[df['CustomerId'] ==
                               selected_customer_id].iloc[0]
    print("Selected Customer:", selected_customer)

    # create two columns to house customer data
    col1, col2 = st.columns(2)
    # add ui elements to display columns
    with col1:

        credit_score = st.number_input("Credit Score",
                                       min_value=300,
                                       max_value=850,
                                       value=int(
                                           selected_customer['CreditScore']))

        location = st.selectbox("Location", ["Spain", "France", "Germany"],
                                index=["Spain", "France", "Germany"
                                       ].index(selected_customer['Geography']))

        gender = st.radio(
            "Gender", ["Male", "Female"],
            index=0 if selected_customer['Gender'] == "Male" else 1)

        age = st.number_input("Age",
                              min_value=18,
                              max_value=100,
                              value=int(selected_customer['Age']))

        tenure = st.number_input("Tenure (years)",
                                 min_value=0,
                                 max_value=50,
                                 value=int(selected_customer['Tenure']))

    with col2:
        balance = st.number_input("Balance",
                                  min_value=0.0,
                                  value=float(selected_customer['Balance']))

        num_products = st.number_input("Number of Products",
                                       min_value=1,
                                       max_value=10,
                                       value=int(
                                           selected_customer['NumOfProducts']))

        has_credit_card = st.checkbox("Has Credit Card",
                                      value=bool(
                                          selected_customer['HasCrCard']))

        is_active_member = st.checkbox(
            "Is Active Member",
            value=bool(selected_customer['IsActiveMember']))

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer['EstimatedSalary']))

    input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                         tenure, balance, num_products,
                                         has_credit_card, is_active_member,
                                         estimated_salary)

    avg_probability = make_predictions(input_df, input_dict)

    explanation = explain_predictions(avg_probability, input_dict,
                                      selected_customer_surname)
    st.markdown("---")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)

    st.markdown("---")

    st.subheader("Personalized Email to Customer")

    email = generate_email(avg_probability, input_dict, explanation,
                           selected_customer_surname)

    st.markdown(email)
