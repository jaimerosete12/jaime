# 📦 Requisitos
# pip install streamlit pybaseball pandas scikit-learn

import streamlit as st
from pybaseball import statcast_pitcher, playerid_lookup, playerid_reverse_lookup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Funciones base compartidas ---
def get_pitcher_id(pitcher_name):
    last, first = pitcher_name.split(" ")[-1], " ".join(pitcher_name.split(" ")[:-1])
    return playerid_lookup(last, first)['key_mlbam'].values[0]

def get_batter_hand(batter_name):
    try:
        last, first = batter_name.split(" ")[-1], " ".join(batter_name.split(" ")[:-1])
        return playerid_reverse_lookup(last + ", " + first)['throws'].values[0]
    except:
        return 'R'

# --- MODELO PRIMER PITCHEO ---
def build_dataset_first(pitcher_id, batter_hand):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data[(data['pitch_number'] == 1) & (data['inning'] == 1)]
    df = df[df['stand'] == batter_hand]
    df = df[['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'description']].dropna()
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    if 'game_date' in df.columns:
        df = df.drop(columns=['game_date'])
    return df

# --- MODELO EN JUEGO ---
def build_dataset_inplay(pitcher_id, batter_hand):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data.dropna(subset=['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'description'])
    df = df[df['stand'] == batter_hand]
    df['release_speed_prev'] = df.groupby('game_pk')['release_speed'].shift(1)
    df = df.dropna(subset=['release_speed_prev'])
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    if 'game_date' in df.columns:
        df = df.drop(columns=['game_date'])
    return df

# --- Entrenamiento y predicción compartidos ---
def train_model(df, casino_line):
    df['target'] = df['release_speed'].apply(lambda x: 'over' if x > casino_line else 'under')
    X = df.drop(columns=['release_speed', 'target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc

def predict(model, feature_columns, input_data):
    input_df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df).max()
    return pred, prob

# --- Streamlit UI ---
st.title("Predicción de Velocidad del Pitcheo: OVER / UNDER")

tab1, tab2 = st.tabs(["Primer Pitcheo", "En Juego (In-Play)"])

with tab1:
    st.header("Predicción para Primer Pitcheo")
    pitcher = st.text_input("Nombre del Pitcher", "Jack Flaherty", key="p1")
    batter = st.text_input("Nombre del Bateador", "Parker Meadows", key="b1")
    casino_line = st.number_input("Línea del Casino (mph)", value=94.0, step=0.1, key="c1")

    if st.button("Predecir Primer Pitcheo"):
        with st.spinner("Procesando datos..."):
            try:
                pitcher_id = get_pitcher_id(pitcher)
                batter_hand = get_batter_hand(batter)
                df = build_dataset_first(pitcher_id, batter_hand)
                model, features, acc = train_model(df, casino_line)

                sample = {
                    'outs_when_up': 0,
                    'pitch_type_FF': 1,
                    'p_throws_R': 1,
                    f'stand_{batter_hand}': 1,
                    'description_called_strike': 1
                }

                pred, prob = predict(model, features, sample)
                st.success(f"Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)")
                st.caption(f"Precisión del modelo: {round(acc*100, 2)}%")
            except Exception as e:
                st.error(f"Error: {e}")

with tab2:
    st.header("Predicción En Juego (In-Play)")
    pitcher = st.text_input("Nombre del Pitcher", "Jack Flaherty", key="p2")
    batter = st.text_input("Nombre del Bateador", "Parker Meadows", key="b2")
    casino_line = st.number_input("Línea del Casino (mph)", value=94.0, step=0.1, key="c2")
    inning = st.number_input("Entrada actual", min_value=1, max_value=9, value=5)
    outs = st.number_input("Outs", min_value=0, max_value=2, value=1)
    release_speed_prev = st.number_input("Velocidad del pitcheo anterior (mph)", value=93.5, step=0.1)
    pitch_type = st.selectbox("Tipo de pitcheo anterior", ['FF', 'SL', 'CH', 'CU'])
    description = st.selectbox("Resultado del pitcheo anterior", ['called_strike', 'ball', 'foul', 'swinging_strike'])

    if st.button("Predecir Siguiente Pitcheo"):
        with st.spinner("Procesando datos..."):
            try:
                pitcher_id = get_pitcher_id(pitcher)
                batter_hand = get_batter_hand(batter)
                df = build_dataset_inplay(pitcher_id, batter_hand)
                model, features, acc = train_model(df, casino_line)

                sample = {
                    'release_speed_prev': release_speed_prev,
                    'outs_when_up': outs,
                    f'pitch_type_{pitch_type}': 1,
                    'p_throws_R': 1,
                    f'stand_{batter_hand}': 1,
                    f'description_{description}': 1
                }

                pred, prob = predict(model, features, sample)
                st.success(f"Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)")
                st.caption(f"Precisión del modelo: {round(acc*100, 2)}%")
            except Exception as e:
                st.error(f"Error al procesar: {e}")
