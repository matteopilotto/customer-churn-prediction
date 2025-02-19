import os
from dotenv import load_dotenv
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import openai
import plotly.graph_objects as go
import json

load_dotenv()

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

xgboost_model = load_model("./models/xgb_model.pkl")
random_forest_model = load_model("./models/rf_model.pkl")
knn_model = load_model("./models/knn_model.pkl")
catboost_model = load_model("./models/catboost_model.pkl")

with open("./models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)


def prepare_input(credit_score, location, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary):

    input_dict  = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Female": 1 if gender == "Male" else 0,
        "Gender_Male": 1 if gender == "Female" else 0
    }

    input_df = pd.DataFrame([input_dict])
    input_df = scaler.transform(input_df)

    return input_df, input_dict


def make_predictions(input_df, input_dict):
    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
        "CatBoost": catboost_model.predict_proba(input_df)[0][1]
    }

    model_probs_list = []

    avg_probability = np.mean(list(probabilities.values()))

    st.markdown("### Model Probabilities")

    for model, prob in probabilities.items():
        model_probs_list.append(f"{model}: {prob:.2%}")
    
    model_probs_str = " | ".join(model_probs_list)
    st.markdown(model_probs_str)
    st.markdown(f"##### Average Probability: {avg_probability:.2%}")

    return avg_probability, probabilities


def explain_prediction(probability, input_dict, surname, churned_customers_stats, active_customers_stats):
    with open("./prompts/prompt_explain_prediction_v2.txt") as f:
        prompt_raw = f.read()

    with open("./prompts/feature_importance.json") as f:
        feature_importance = json.load(f)
        feature_importance = json.dumps(feature_importance, indent=4)

    probability = round(probability * 100, 2)
    user_profile = json.dumps(input_dict, indent=4)

    prompt = prompt_raw.format(
        surname=surname,
        prob=probability,
        user_profile=user_profile,
        feature_importance=feature_importance,
        # churned_customers_stats=churned_customers_stats,
        # active_customers_stats=active_customers_stats
    ).strip()

    raw_response = client.chat.completions.create(
        model="llama-3.2-3b-preview",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
    with open("./prompts/prompt_generate_email_v2.txt", "r") as f:
        prompt_raw = f.read()

    prob = round(probability * 100, 2)

    prompt = prompt_raw.format(
        surname=surname,
        user_profile=input_dict,
        explanation=explanation
    )

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )

    return raw_response.choices[0].message.content


def create_gauge_chart(probability):
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"
    

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={
                "x": [0, 1],
                "y": [0, 1]
            },
            title={
                "text": "Churn Probability",
                "font": {
                    "size": 24,
                    "color": "white"
                }
            },
            number={"font": {
                "size": 40,
                "color": "white"
            }},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "white"
                },
                "bar": {
                    "color": color
                },
                "bgcolor": "rgba(0, 0, 0, 0)",
                "borderwidth": 2,
                "bordercolor": "white",
                "steps": [{
                    "range": [0, 30],
                    "color": "rgba(0, 255, 0, 0.3)"
                }, {
                    "range": [30, 60],
                    "color": "rgba(255, 255, 0, 0.3)"
                }, {
                    "range": [60, 100],
                    "color": "rgba(255, 0, 0, 0.3)"
                }],
                "threshold": {
                    "line": {
                        "color": "white",
                        "width": 4
                    },
                    "thickness": 0.75,
                    "value": 100
                }
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "white"},
        width=400,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig

def create_model_probability_chart(probabilites):
    models = list(probabilites.keys())
    probs = list(probabilites.values())

    fig = go.Figure(data=[
        go.Bar(
            y=models,
            x=probs,
            orientation="h",
            text=[f"{p:.2%}" for p in probs],
            textposition="auto"
        )
    ])

    fig.update_layout(
        title="Churn Probability by Model",
        yaxis_title="Models",
        xaxis_title="Probability",
        xaxis=dict(tickformat=".0%", range=[0, 1]),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig