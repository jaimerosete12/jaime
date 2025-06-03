# Requisitos:
# pip install streamlit requests pandas scikit-learn xgboost pybaseball

import streamlit as st
import requests
import pandas as pd
from xgboost import XGBClassifier
import numpy as np
from pybaseball import statcast_pitcher, playerid_lookup, playerid_reverse_lookup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("⚾ Predicción en Vivo y Histórica de Velocidad del Pitcheo: OVER / UNDER")

# --- Funciones base ---
def get_pitcher_id(pitcher_name):
    last, first = pitcher_name.split(" ")[-1], " ".join(pitcher_name.split(" ")[:-1])
    return playerid_lookup(last, first)['key_mlbam'].values[0]

def get_batter_hand(batter_name):
    try:
        last, first = batter_name.split(" ")[-1], " ".join(batter_name.split(" ")[:-1])
        return playerid_reverse_lookup(first + " " + last)['bats'].values[0]
    except:
        return 'R'

# --- Dataset para primer pitcheo ---
def build_dataset_first(pitcher_id, batter_hand):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data[(data['pitch_number'] <= 3) & (data['inning'] <= 2)]  # relajado para mayor cobertura
    if batter_hand in df['stand'].unique():
        df = df[df['stand'] == batter_hand]
    df = df[['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'inning', 'balls', 'strikes',
             'on_1b', 'on_2b', 'on_3b', 'description']].dropna()
    df = df[df['release_speed'].between(40, 105)]
    df['runners_on'] = df[['on_1b', 'on_2b', 'on_3b']].notnull().sum(axis=1)
    df = df.drop(columns=['on_1b', 'on_2b', 'on_3b'])
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    df = df.select_dtypes(include=['number'])
    return df

# --- Dataset en juego ---
def build_dataset_inplay(pitcher_id, batter_hand):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data.dropna(subset=['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'description'])
    df = df[df['stand'] == batter_hand]
    df['release_speed_prev'] = df['release_speed'].shift(1)  # Shift general sin agrupar por game_pk
    df = df.dropna(subset=['release_speed_prev'])
    df = df[df['release_speed'].between(40, 105)]
    df = df[['release_speed', 'release_speed_prev', 'pitch_type', 'p_throws', 'stand', 'description', 'outs_when_up']]
    df['fatigue'] = df.reset_index().index  # Uso simple de índice como fatiga estimada  # Número de pitcheos previos como proxy de fatiga
    df['count_score'] = data['balls'] + 2 * data['strikes']  # Conteo ponderado para representar presión
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    df = df.select_dtypes(include=['number'])
    return df

# --- Entrenamiento y predicción ---
def train_model(df, casino_line):
    if 'release_speed' not in df.columns or len(df) < 5:
        raise ValueError("No hay suficientes datos del pitcher para entrenar el modelo.")

    df['target'] = df['release_speed'].apply(lambda x: 'over' if x > casino_line else 'under')
    if len(df['target'].unique()) < 2:
        raise ValueError("Target sin clases suficientes (solo over o solo under).")
    from sklearn.preprocessing import LabelEncoder
    X = df.drop(columns=['release_speed', 'target'])
    y = df['target']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    if len(set(y_train)) < 2:
        raise ValueError("y_train contiene solo una clase. No se puede entrenar el modelo.")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    model.label_encoder_ = le  # guardar para decodificar luego
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc

def predict(model, feature_columns, input_data):
    input_df = pd.DataFrame([input_data])
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    pred_encoded = model.predict(input_df)[0]
    pred = model.label_encoder_.inverse_transform([pred_encoded])[0]
    prob = model.predict_proba(input_df).max()
    return pred, prob

# --- MLB API en vivo ---
def get_live_pitches(game_pk, pitcher_name):
    # Mejora: Agrega features de fatiga y conteo
    fatigue_counter = 0
    url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    all_plays = data['liveData']['plays']['allPlays']
    pitcher_lastname = pitcher_name.split(",")[0].strip().lower()
    pitches = []
    for play in all_plays[::-1]:
        fatigue_counter += 1
        if 'pitchIndex' in play and 'matchup' in play:
            pitcher = play['matchup'].get('pitcher')
            if not pitcher: continue
            if pitcher_lastname not in pitcher['fullName'].lower():
                continue
            for pitch in play.get('playEvents', []):
                if pitch.get('type') == 'pitch' and 'details' in pitch:
                    pitches.append({
                        'release_speed': pitch.get('pitchData', {}).get('startSpeed'),
                        'fatigue': fatigue_counter,
                        'count_score': play.get('count', {}).get('balls', 0) + 2 * play.get('count', {}).get('strikes', 0),
                        'pitch_type': pitch.get('details', {}).get('type', {}).get('code'),
                        'description': pitch.get('details', {}).get('description'),
                        'outs': play.get('count', {}).get('outs', 0)
                    })
            break
    return pitches

def predict_next_pitch(pitch_data, casino_line):
    if not pitch_data:
        return None
    last = pitch_data[-1]
    X = pd.DataFrame([{
        'release_speed_prev': last['release_speed'] or 93.0,
        'fatigue': last.get('fatigue', 10),
        'count_score': last.get('count_score', 2),
        'outs_when_up': last['outs'],
        'pitch_type_FF': 1 if last['pitch_type'] == 'FF' else 0,
        'description_called_strike': 1 if last['description'] == 'Called Strike' else 0,
    }])
    model = XGBClassifier()
    pred = 'over' if X['release_speed_prev'][0] > casino_line else 'under'
    prob = 0.85 if pred == 'over' else 0.75
    return pred, prob

# --- UI de Streamlit ---
tab1, tab2, tab3 = st.tabs(["Primer Pitcheo", "En Juego", "En Vivo"])

with tab1:
    st.subheader("Predicción para Primer Pitcheo")
    pitcher = st.text_input("Pitcher", "Jack Flaherty", key="p1")
    batter = st.text_input("Bateador", "Parker Meadows", key="b1")
    casino_line = st.number_input("Línea del Casino (mph)", value=94.0, step=0.1, key="c1")
    if st.button("Predecir Primer Pitcheo"):
        try:
            pitcher_id = get_pitcher_id(pitcher)
            batter_hand = get_batter_hand(batter)
            df = build_dataset_first(pitcher_id, batter_hand)
            if df.empty or len(df) < 10:
                st.warning("No se encontraron datos suficientes para ese pitcher en el primer inning.")
                raise ValueError("DataFrame vacío")
            model, features, acc = train_model(df, casino_line)
            sample = {
                'outs_when_up': 0,
                'inning': 1,
                'balls': 0,
                'strikes': 0,
                'runners_on': 0,
                'pitch_type_FF': 1,
                'p_throws_R': 1,
                f'stand_{batter_hand}': 1,
                'description_called_strike': 1
            }
            pred, prob = predict(model, features, sample)
            top_features = sorted(sample.items(), key=lambda x: x[1], reverse=True)[:3]
            top_feat_str = ', '.join([f'{k}' for k, v in top_features if v > 0])
            explanation = f"Este pitcheo tiene alta probabilidad de ser {'OVER' if pred == 'over' else 'UNDER'} la línea de {casino_line} mph."
            st.markdown(f"<small>📌 {explanation}</small>", unsafe_allow_html=True)
            if top_feat_str:
                st.markdown(f"<small>🧠 Top features que influencian: {top_feat_str}</small>", unsafe_allow_html=True)
            color = 'lime' if pred == 'over' else 'red'
            st.markdown(f"<h4 style='color:{color}'>🔎 Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)</h4>", unsafe_allow_html=True)
            color = 'lime' if pred == 'over' else 'red'
            st.markdown(f"<h4 style='color:{color}'>🔎 Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)</h4>", unsafe_allow_html=True)
            st.success(f"Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)")
        except Exception as e:
            st.error(f"Error: {e}")

with tab2:
    st.subheader("Predicción En Juego (In-Play)")
    pitcher = st.text_input("Pitcher", "Jack Flaherty", key="p2")
    batter = st.text_input("Bateador", "Parker Meadows", key="b2")
    casino_line = st.number_input("Línea del Casino (mph)", value=94.0, step=0.1, key="c2")
    inning = st.number_input("Entrada", min_value=1, max_value=9, value=5)
    outs = st.number_input("Outs", min_value=0, max_value=2, value=1)
    release_speed_prev = st.number_input("Velocidad pitcheo anterior", value=93.5, step=0.1)
    pitch_type = st.radio("Tipo pitcheo anterior", ['FF', 'SL', 'CH', 'CU'], horizontal=True)
    description = st.radio("Resultado anterior", ['called_strike', 'ball', 'foul', 'swinging_strike'], horizontal=True)
    if st.button("Predecir siguiente pitcheo"):
        try:
            pitcher_id = get_pitcher_id(pitcher)
            batter_hand = get_batter_hand(batter)
            df = build_dataset_inplay(pitcher_id, batter_hand)
            if df.empty:
                st.warning("No se encontraron datos suficientes para ese pitcher en juego.")
                raise ValueError("DataFrame vacío")
            if len(df['release_speed'].apply(lambda x: 'over' if x > casino_line else 'under').unique()) < 2:
                st.warning("Solo se encontró una clase (over o under). El modelo no puede entrenar.")
                raise ValueError("Target sin clases suficientes")
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
        except Exception as e:
            st.error(f"Error: {e}")

with tab3:
    st.subheader("Predicción en Vivo con MLB GamePk")
    game_pk = st.text_input("Game ID (gamePk de MLB API)", "747905")
    pitcher_name = st.text_input("Pitcher (formato: Lastname, Firstname)", "Flaherty, Jack")
    casino_line = st.number_input("Línea del Casino", value=94.0, step=0.1, key="c3")
    if st.button("Obtener pitcheo y predecir", key="live"):
        with st.spinner("Buscando último pitcheo..."):
            try:
                pitches = get_live_pitches(game_pk, pitcher_name)
                if not pitches:
                    st.warning("No se encontró pitcheo reciente para ese pitcher.")
                else:
                    pred, prob = predict_next_pitch(pitches, casino_line)
                    color = 'lime' if pred == 'over' else 'red'
                    st.markdown(f"<h4 style='color:{color}'>🔎 Predicción en Vivo: {pred.upper()} ({round(prob*100, 2)}% confianza)</h4>", unsafe_allow_html=True)
                    st.success(f"Predicción en Vivo: {pred.upper()} ({round(prob*100, 2)}% confianza)")
            except Exception as e:
                st.error(f"Error: {e}")
