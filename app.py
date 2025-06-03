# 📦 Requisitos
# pip install streamlit pybaseball pandas scikit-learn

import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup, playerid_reverse_lookup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Funciones base ---
def get_pitcher_id(pitcher_name):
    last, first = pitcher_name.split(" ")[-1], " ".join(pitcher_name.split(" ")[:-1])
    return playerid_lookup(last, first)['key_mlbam'].values[0]

def get_batter_hand(batter_name):
    try:
        last, first = batter_name.split(" ")[-1], " ".join(batter_name.split(" ")[:-1])
        return playerid_reverse_lookup(last + ", " + first)['throws'].values[0]
    except:
        return 'R'

def build_dataset(pitcher_id, batter_hand, start_date='2022-03-01', end_date='2024-11-01'):
    data = statcast_pitcher(start_date, end_date, pitcher_id)
    df = data[(data['pitch_number'] == 1) & (data['inning'] == 1)]
    df = df[df['stand'] == batter_hand]
    df = df[['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'description']].dropna()
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    return df

def train_model(df, casino_line):
    df['target'] = df['release_speed'].apply(lambda x: 'over' if x > casino_line else 'under')
    X = df.drop(columns=['release_speed', 'target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc

def predict_first_pitch(model, feature_columns, latest_pitch_data):
    input_df = pd.DataFrame([latest_pitch_data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max()
    return pred, prob

# --- Streamlit UI ---
st.title("Predicción Over/Under del Primer Pitcheo")

pitcher = st.text_input("Nombre del Pitcher", "Jack Flaherty")
batter = st.text_input("Nombre del Bateador", "Parker Meadows")
casino_line = st.number_input("Línea del Casino (mph)", value=94.0, step=0.1)

if st.button("Predecir"):
    with st.spinner("Obteniendo datos y entrenando modelo..."):
        try:
            pitcher_id = get_pitcher_id(pitcher)
            batter_hand = get_batter_hand(batter)
            df = build_dataset(pitcher_id, batter_hand)
            model, feature_columns, acc = train_model(df, casino_line)

            latest_pitch = {
                'outs_when_up': 0,
                'pitch_type_FF': 1,
                'p_throws_R': 1,
                f'stand_{batter_hand}': 1,
                'description_called_strike': 1
            }

            pred, prob = predict_first_pitch(model, feature_columns, latest_pitch)

            st.success(f"Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)")
            st.caption(f"Precisión histórica del modelo: {round(acc*100, 2)}%")

        except Exception as e:
            st.error(f"Error al procesar: {e}")




