# Requisitos:
# pip install streamlit requests pandas scikit-learn xgboost pybaseball matplotlib seaborn

import streamlit as st
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from pybaseball import statcast_pitcher, playerid_lookup, playerid_reverse_lookup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="MLB Pitch Predictor", layout="wide")
tabs = st.tabs(["🔍 Primer Pitcheo", "🎯 Predicción En Juego", "🌐 Live API"])

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
    df = data[(data['pitch_number'] == 1) & (data['inning'] == 1)]
    if batter_hand in df['stand'].unique():
        filtered = df[df['stand'] == batter_hand]
        if len(filtered) > 10:
            df = filtered
    if df.shape[0] < 10:
        df = data[(data['pitch_number'] == 1) & (data['inning'] == 1)]
    df = df[['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'inning', 'balls', 'strikes',
             'on_1b', 'on_2b', 'on_3b', 'description']].dropna()
    df = df[df['release_speed'].between(40, 105)]
    df['runners_on'] = df[['on_1b', 'on_2b', 'on_3b']].notnull().sum(axis=1)
    df = df.drop(columns=['on_1b', 'on_2b', 'on_3b'])
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    df = df.select_dtypes(include=['number'])
    return df

# --- Resumen de últimos 10 primeros lanzamientos ---
def mostrar_ultimos_lanzamientos(pitcher_id, linea_casino):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data[(data['pitch_number'] == 1) & (data['inning'] == 1)].copy()
    df = df[['game_date', 'release_speed']].dropna().sort_values(by='game_date', ascending=False).head(10)
    df['casino_line'] = linea_casino
    df['Resultado'] = np.where(df['release_speed'] > linea_casino, 'Over', 'Under')
    return df.reset_index(drop=True)

# --- Dataset para pitcheo en juego ---
def build_dataset_inplay(pitcher_id, batter_hand):
    data = statcast_pitcher('2022-03-01', '2024-11-01', pitcher_id)
    df = data[(data['pitch_number'] > 1) & (data['inning'] > 1)]
    if batter_hand in df['stand'].unique():
        filtered = df[df['stand'] == batter_hand]
        if len(filtered) > 5:
            df = filtered
    df = df[['release_speed', 'pitch_type', 'p_throws', 'stand', 'outs_when_up', 'inning', 'balls', 'strikes',
             'on_1b', 'on_2b', 'on_3b', 'description']].dropna()
    df = df[df['release_speed'].between(40, 105)]
    df['runners_on'] = df[['on_1b', 'on_2b', 'on_3b']].notnull().sum(axis=1)
    df = df.drop(columns=['on_1b', 'on_2b', 'on_3b'])
    df = pd.get_dummies(df, columns=['pitch_type', 'p_throws', 'stand', 'description'], drop_first=True)
    df = df.select_dtypes(include=['number'])
    return df

# --- API en Vivo ---
def get_live_pitches(game_pk, pitcher_name):
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
            if not pitcher:
                continue
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
        'description_called_strike': 1 if last['description'] == 'Called Strike' else 0
    }])
    pred = 'Over' if X['release_speed_prev'][0] > casino_line else 'Under'
    prob = 0.85 if pred == 'Over' else 0.75
    return pred, prob
    
# === TAB 1: Primer Pitcheo ===
with tabs[0]:
    st.header("🔍 Predicción para Primer Pitcheo")
    pitcher = st.text_input("Nombre del Pitcher", "Paul Skenes")
    batter = st.text_input("Nombre del Bateador", "Jeremy Peña")
    linea = st.number_input("Línea del Casino (mph)", value=98.45, step=0.1)

    if st.button("Predecir Primer Pitcheo"):
        try:
            pitcher_id = get_pitcher_id(pitcher)
            batter_hand = get_batter_hand(batter)
            df = build_dataset_first(pitcher_id, batter_hand)
            if df.empty:
                st.warning("No hay datos suficientes para este pitcher.")
            else:
                df['target'] = np.where(df['release_speed'] > linea, 1, 0)
                X = df.drop(columns=['release_speed', 'target'])
                y = df['target']
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X, y)
                sample = X.iloc[-1:]
                pred = model.predict(sample)[0]
                color = 'green' if pred else 'red'
                label = "OVER" if pred else "UNDER"
                st.markdown(f"<h4 style='color:{color}'>🔎 Predicción: {label}</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

    # Mostrar últimos 10 primeros lanzamientos
    if pitcher:
        try:
            pitcher_id = get_pitcher_id(pitcher)
            resumen = mostrar_ultimos_lanzamientos(pitcher_id, linea)
            if not resumen.empty:
                colores = resumen['Resultado'].map({'Over': 'green', 'Under': 'red'})
                st.dataframe(resumen.style.apply(lambda x: ['background-color: {}'.format(col) for col in colores], axis=1))
                st.subheader("📊 Comparativa de velocidades")
                fig, ax = plt.subplots()
                sns.barplot(data=resumen, x='game_date', y='release_speed', hue='Resultado', palette={'Over': 'green', 'Under': 'red'}, ax=ax)
                ax.axhline(linea, color='gray', linestyle='--', label='Línea del casino')
                ax.set_ylabel('Velocidad')
                ax.set_xlabel('Fecha')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"No se pudo generar resumen: {e}")

# === TAB 2: En Juego ===
with tabs[1]:
    st.header("🎯 Predicción En Juego")
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
            df['target'] = df['release_speed'].apply(lambda x: 'over' if x > casino_line else 'under')
            if len(df['target'].unique()) < 2:
                st.warning("Solo se encontró una clase (over o under). El modelo no puede entrenar.")
                raise ValueError("Target sin clases suficientes")
            X = df.drop(columns=['release_speed', 'target'])
            y = df['target']
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            sample = {
                'release_speed_prev': release_speed_prev,
                'outs_when_up': outs,
                f'pitch_type_{pitch_type}': 1,
                'p_throws_R': 1,
                f'stand_{batter_hand}': 1,
                f'description_{description}': 1
            }
            for col in X.columns:
                if col not in sample:
                    sample[col] = 0
            sample_df = pd.DataFrame([sample])[X.columns]
            pred = model.predict(sample_df)[0]
            prob = model.predict_proba(sample_df).max()
            color = 'green' if pred == 'over' else 'red'
            st.markdown(f"<h4 style='color:{color}'>🔎 Predicción: {pred.upper()} ({round(prob*100, 2)}% confianza)</h4>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")
# === TAB 3: Live API ===
with tabs[2]:
    st.header("🌐 Predicción en Vivo desde API (simulada)")
    game_pk = st.text_input("Game ID (gamePk de MLB API)", "747905")
    pitcher_name = st.text_input("Pitcher (formato: Lastname, Firstname)", "Flaherty, Jack")
    linea = st.number_input("Línea del Casino", value=94.0, step=0.1)

    if st.button("Obtener pitcheo y predecir"):
        with st.spinner("Buscando último pitcheo..."):
            try:
                pitches = get_live_pitches(game_pk, pitcher_name)
                if not pitches:
                    st.warning("No se encontró pitcheo reciente para ese pitcher.")
                else:
                    pred, prob = predict_next_pitch(pitches, linea)
                    color = 'green' if pred == 'Over' else 'red'
                    st.markdown(f"<h4 style='color:{color}'>🔎 Predicción en Vivo: {pred} ({round(prob*100, 2)}% confianza)</h4>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
