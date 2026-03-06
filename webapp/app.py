import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
import shutil
from datetime import datetime
from sklearn.metrics import fbeta_score

# ─────────────────────────────────────────────
#  CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight — Prédiction de Churn",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  CSS CUSTOM
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --border:    #1f2d45;
    --accent:    #00d4aa;
    --accent2:   #ff6b6b;
    --accent3:   #f4c430;
    --text:      #e2e8f0;
    --muted:     #64748b;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

h1, h2, h3 { font-family: 'Syne', sans-serif; }

[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

.badge-churn {
    display: inline-block;
    background: linear-gradient(135deg, #ff6b6b22, #ff6b6b44);
    border: 1px solid var(--accent2);
    border-radius: 100px;
    padding: 6px 20px;
    color: var(--accent2);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-stay {
    display: inline-block;
    background: linear-gradient(135deg, #00d4aa22, #00d4aa44);
    border: 1px solid var(--accent);
    border-radius: 100px;
    padding: 6px 20px;
    color: var(--accent);
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.proba-big {
    font-family: 'Syne', sans-serif;
    font-size: 4rem;
    font-weight: 800;
    line-height: 1;
    background: linear-gradient(135deg, #00d4aa, #f4c430);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
    border-radius: 100px !important;
}

.metric-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.metric-box .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--accent);
}
.metric-box .lbl {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

hr { border-color: var(--border); }

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] > div > div {
    background: var(--bg) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    margin-top: 20px;
}

.alert-retrain {
    background: linear-gradient(135deg, #f4c43011, #f4c43022);
    border: 1px solid var(--accent3);
    border-radius: 12px;
    padding: 16px 20px;
    color: var(--accent3);
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CHEMINS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedbacks.json")


# ─────────────────────────────────────────────
#  CHARGEMENT DES ARTIFACTS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    artifacts = {}
    paths = {
        'model':     os.path.join(BASE_DIR, 'artifacts/model.pickle'),
        'scaler':    os.path.join(BASE_DIR, 'artifacts/scaler.pickle'),
        'columns':   os.path.join(BASE_DIR, 'artifacts/input_columns.pickle'),
        'features':  os.path.join(BASE_DIR, 'artifacts/pca.pickle'),
        'threshold': os.path.join(BASE_DIR, 'artifacts/optimal_threshold.pickle'),
        'info':      os.path.join(BASE_DIR, 'artifacts/model_info.pickle'),
    }
    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'rb') as f:
                artifacts[key] = pickle.load(f)
        else:
            artifacts[key] = None
    return artifacts


# ─────────────────────────────────────────────
#  PRÉDICTIONS
# ─────────────────────────────────────────────
def predict_single(data_dict, artifacts):
    """Prédiction pour un client unique."""
    df = pd.DataFrame([data_dict])
    cols = artifacts['columns']
    df = df[cols]
    X_scaled = artifacts['scaler'].transform(df)
    X_selected = X_scaled[:, artifacts['features']]
    proba = artifacts['model'].predict_proba(X_selected)[:, 1][0]
    threshold = artifacts['threshold'] if artifacts['threshold'] else 0.5
    pred = int(proba >= threshold)
    return pred, proba


def predict_batch(df_raw, artifacts):
    """Prédiction batch pour un fichier CSV."""
    cols = artifacts['columns']
    df = df_raw[cols].copy()
    X_scaled = artifacts['scaler'].transform(df)
    X_selected = X_scaled[:, artifacts['features']]
    probas = artifacts['model'].predict_proba(X_selected)[:, 1]
    threshold = artifacts['threshold'] if artifacts['threshold'] else 0.5
    preds = (probas >= threshold).astype(int)
    return preds, probas


# ─────────────────────────────────────────────
#  GESTION DES FEEDBACKS
# ─────────────────────────────────────────────
def load_feedbacks():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return []

def save_feedback(entry):
    feedbacks = load_feedbacks()
    feedbacks.append(entry)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedbacks, f, indent=2)
    return feedbacks

def check_retrain_needed(threshold=20):
    feedbacks = load_feedbacks()
    wrong = [f for f in feedbacks if f.get('correct') == False]
    return len(feedbacks) >= threshold, len(feedbacks), len(wrong)


# ─────────────────────────────────────────────
#  RE-ENTRAINEMENT RÉEL
# ─────────────────────────────────────────────
def retrain_model():
    """Ré-entraîne le modèle Stacking avec les feedbacks corrigés."""
    from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier, BaggingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    from category_encoders import TargetEncoder

    feedbacks = load_feedbacks()

    # 1. Vérification feedbacks avec vraie étiquette
    labeled = [f for f in feedbacks if f.get('true_label') is not None and f.get('data')]
    if len(labeled) < 5:
        return False, f"❌ Seulement {len(labeled)} feedbacks avec étiquette. Minimum 5 requis."

    # 2. Chargement des données originales
    data_path = os.path.join(BASE_DIR, 'data', 'Churn_Modelling.csv')
    if not os.path.exists(data_path):
        return False, "❌ Fichier Churn_Modelling.csv introuvable dans data/"

    df_original = pd.read_csv(data_path)
    X_original  = df_original.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1)
    Y_original  = df_original['Exited']

    # 3. Construction des nouvelles données depuis les feedbacks
    new_rows, new_labels = [], []
    for f in labeled:
        data = f['data']
        if data:
            new_rows.append({
                'CreditScore':     data.get('CreditScore',     650),
                'Geography':       data.get('Geography',       'France'),
                'Gender':          data.get('Gender',          'Male'),
                'Age':             data.get('Age',             40),
                'Tenure':          data.get('Tenure',          3),
                'Balance':         data.get('Balance',         0),
                'NumOfProducts':   data.get('NumOfProducts',   1),
                'HasCrCard':       data.get('HasCrCard',       1),
                'IsActiveMember':  data.get('IsActiveMember',  1),
                'EstimatedSalary': data.get('EstimatedSalary', 60000),
            })
            new_labels.append(f['true_label'])

    # 4. Fusion données originales + feedbacks
    if new_rows:
        df_new     = pd.DataFrame(new_rows)
        Y_new      = pd.Series(new_labels)
        X_combined = pd.concat([X_original, df_new], ignore_index=True)
        Y_combined = pd.concat([Y_original, Y_new],  ignore_index=True)
    else:
        X_combined = X_original
        Y_combined = Y_original

    # 5. Nouveau preprocessor
    categorical_features = ['Geography', 'Gender']
    numeric_features     = ['CreditScore', 'Age', 'Tenure', 'Balance',
                            'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

    new_preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(),                                  numeric_features),
        ('cat', TargetEncoder(smoothing=1.0, min_samples_leaf=1), categorical_features)
    ])
    X_transformed = new_preprocessor.fit_transform(X_combined, Y_combined)

    # 6. Features sélectionnées
    arts = load_artifacts()
    selected_features = arts.get('features')
    if selected_features is None:
        return False, "❌ Impossible de charger la sélection de features."

    X_selected = X_transformed[:, selected_features]

    # 7. Ré-entraînement du Stacking
    estimators = [
        ('GradBoost', GradientBoostingClassifier(n_estimators=200, random_state=42)),
        ('Bagging',   BaggingClassifier(n_estimators=200,          random_state=42)),
        ('MLP',       MLPClassifier(hidden_layer_sizes=(50, 25),   random_state=42, max_iter=400)),
    ]
    new_stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        stack_method='predict_proba',
        n_jobs=-1
    )

    # Split pour évaluation
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X_selected, Y_combined, test_size=0.2, stratify=Y_combined, random_state=42
    )
    new_stacking.fit(X_tr, Y_tr)

    # 8. Recalcul du seuil optimal
    y_proba = new_stacking.predict_proba(X_te)[:, 1]
    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(Y_te, y_proba)

    f2_scores = []
    for p, r in zip(precision_arr, recall_arr):
        if p + r > 0:
            f2_scores.append((1 + 4) * p * r / (4 * p + r))
        else:
            f2_scores.append(0)

    best_idx      = int(np.argmax(f2_scores))
    new_threshold = float(thresholds_arr[best_idx]) if best_idx < len(thresholds_arr) else 0.5
    y_pred_new    = (y_proba >= new_threshold).astype(int)
    new_f2        = fbeta_score(Y_te, y_pred_new, beta=2, zero_division=0)

    # 9. Backup et sauvegarde des nouveaux artifacts
    artifacts_dir = os.path.join(BASE_DIR, 'artifacts')
    backup_dir    = os.path.join(artifacts_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    for fname in ['model.pickle', 'scaler.pickle', 'optimal_threshold.pickle', 'model_info.pickle']:
        src = os.path.join(artifacts_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(backup_dir, f"{timestamp}_{fname}"))

    with open(os.path.join(artifacts_dir, 'model.pickle'), 'wb') as f:
        pickle.dump(new_stacking, f)

    with open(os.path.join(artifacts_dir, 'scaler.pickle'), 'wb') as f:
        pickle.dump(new_preprocessor, f)

    with open(os.path.join(artifacts_dir, 'optimal_threshold.pickle'), 'wb') as f:
        pickle.dump(new_threshold, f)

    old_info = arts.get('info') or {}
    new_info = {
        'name':              'Stacking (GradBoost + Bagging + MLP)',
        'accuracy':          old_info.get('accuracy', 0),
        'auc':               old_info.get('auc', 0),
        'f2_score':          new_f2,
        'optimal_threshold': new_threshold,
        'last_retrain':      timestamp,
        'nb_feedbacks_used': len(labeled),
    }
    with open(os.path.join(artifacts_dir, 'model_info.pickle'), 'wb') as f:
        pickle.dump(new_info, f)

    # 10. Réinitialisation du cache Streamlit
    st.cache_resource.clear()

    return True, (
        f"✅ Modèle ré-entraîné avec succès !\n\n"
        f"• {len(labeled)} feedbacks intégrés\n"
        f"• Nouveau F2-Score : {new_f2:.4f}\n"
        f"• Nouveau seuil   : {new_threshold:.4f}\n"
        f"• Backup sauvegardé dans artifacts/backups/"
    )


# ─────────────────────────────────────────────
#  CHARGEMENT INITIAL
# ─────────────────────────────────────────────
artifacts = load_artifacts()
model_ok  = artifacts['model'] is not None


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; color:#00d4aa;">
            🔮 ChurnSight
        </div>
        <div style="font-size:0.75rem; color:#64748b; margin-top:4px;">
            Système de prédiction de départ client
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("Navigation", [
        "🎯  Prédiction Manuelle",
        "📂  Import CSV",
        "📊  Dashboard Feedbacks",
    ], label_visibility="collapsed")

    st.markdown("---")

    # Statut modèle
    st.markdown('<div class="section-title">Statut du modèle</div>', unsafe_allow_html=True)
    if model_ok:
        info = artifacts.get('info') or {}
        last_retrain = info.get('last_retrain', 'Jamais')
        st.markdown(f"""
        <div style="font-size:0.8rem; color:#64748b; line-height:2;">
            <span style="color:#00d4aa;">●</span> Modèle chargé<br>
            <b style="color:#e2e8f0;">{info.get('name','Stacking')}</b><br>
            AUC   : <b style="color:#00d4aa;">{info.get('auc', 0):.4f}</b><br>
            F2    : <b style="color:#00d4aa;">{info.get('f2_score', 0):.4f}</b><br>
            Seuil : <b style="color:#f4c430;">{artifacts.get('threshold', 0.5):.4f}</b><br>
            Dernier re-train : <b style="color:#e2e8f0;">{last_retrain}</b>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("⚠️ Artifacts introuvables")

    st.markdown("---")

    # Compteur feedbacks
    needs_retrain, total_fb, wrong_fb = check_retrain_needed(20)
    st.markdown('<div class="section-title">Feedbacks collectés</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#64748b; line-height:2;">
        Total       : <b style="color:#e2e8f0;">{total_fb}</b><br>
        Corrections : <b style="color:#ff6b6b;">{wrong_fb}</b><br>
        Prochain re-train : <b style="color:#f4c430;">{max(0, 20 - total_fb)} feedbacks restants</b>
    </div>
    """, unsafe_allow_html=True)

    if needs_retrain:
        st.markdown("""
        <div class="alert-retrain">
            ⚡ 20 feedbacks atteints !<br>
            Re-entraînement disponible.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🔄 Re-entraîner le modèle", use_container_width=True):
            with st.spinner("⏳ Re-entraînement en cours (~2-3 min)..."):
                success, msg = retrain_model()
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.warning(msg)


# ─────────────────────────────────────────────
#  PAGE 1 — PRÉDICTION MANUELLE
# ─────────────────────────────────────────────
if "Manuelle" in page:

    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-bottom:4px;">
        Prédiction Client
    </h1>
    <p style="color:#64748b; margin-bottom:32px;">
        Renseignez les informations du client pour obtenir une prédiction instantanée.
    </p>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Informations client</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            credit_score = st.number_input("Credit Score",       300,    900,    650,      help="Score de crédit du client")
            age          = st.number_input("Âge",                18,     100,    40)
            tenure       = st.number_input("Ancienneté (années)",0,      10,     3)
            balance      = st.number_input("Solde (€)",          0.0,    300000.0, 50000.0, step=1000.0)
        with c2:
            num_products = st.selectbox("Nb. de produits",       [1, 2, 3, 4],   index=0)
            has_cr_card  = st.selectbox("Carte de crédit",       ["Oui", "Non"], index=0)
            is_active    = st.selectbox("Membre actif",          ["Oui", "Non"], index=0)
            salary       = st.number_input("Salaire estimé (€)", 0.0,    300000.0, 60000.0, step=1000.0)

        c3, c4 = st.columns(2)
        with c3:
            geography = st.selectbox("Pays",  ["France", "Germany", "Spain"])
        with c4:
            gender    = st.selectbox("Genre", ["Male", "Female"])

        st.markdown('</div>', unsafe_allow_html=True)
        predict_btn = st.button("🔮 Lancer la prédiction", use_container_width=True, type="primary")

    with col_result:
        if predict_btn and model_ok:
            data = {
                'CreditScore':     credit_score,
                'Geography':       geography,
                'Gender':          gender,
                'Age':             age,
                'Tenure':          tenure,
                'Balance':         balance,
                'NumOfProducts':   num_products,
                'HasCrCard':       1 if has_cr_card == "Oui" else 0,
                'IsActiveMember':  1 if is_active   == "Oui" else 0,
                'EstimatedSalary': salary,
            }
            try:
                pred, proba = predict_single(data, artifacts)
                st.session_state['last_pred']  = pred
                st.session_state['last_proba'] = proba
                st.session_state['last_data']  = data
                st.session_state['fb_given']   = False
                st.session_state['fb_wrong']   = False
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")

        # ── Affichage résultat ──
        if 'last_pred' in st.session_state:
            pred  = st.session_state['last_pred']
            proba = st.session_state['last_proba']

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">🎯 Résultat</div>', unsafe_allow_html=True)

            # Badge
            if pred == 1:
                st.markdown('<span class="badge-churn">⚠️ Risque de départ</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-stay">✅ Client fidèle</span>', unsafe_allow_html=True)

            # Proba + Prédiction côte à côte
            statut_txt = "⚠️ DÉPART" if pred == 1 else "✅ FIDÈLE"
            statut_col = "#ff6b6b"   if pred == 1 else "#00d4aa"

            st.markdown(f"""
            <div style="margin:20px 0 8px 0; display:flex; align-items:flex-end; gap:32px;">
                <div>
                    <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em;">
                        Probabilité de départ
                    </div>
                    <div class="proba-big">{proba:.1%}</div>
                </div>
                <div style="margin-bottom:10px;">
                    <div style="font-size:0.7rem; color:#64748b; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                        Prédiction
                    </div>
                    <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:{statut_col};">
                        {statut_txt}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(proba))
            st.markdown('</div>', unsafe_allow_html=True)

            # ── FEEDBACK ──
            if not st.session_state.get('fb_given', False):
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">💬 Cette prédiction est-elle correcte ?</div>', unsafe_allow_html=True)
                st.markdown('<p style="font-size:0.8rem; color:#64748b; margin-bottom:12px;">Votre retour améliore le modèle.</p>', unsafe_allow_html=True)

                fb1, fb2 = st.columns(2)
                with fb1:
                    if st.button("👍  Oui, correcte", use_container_width=True):
                        entry = {
                            'timestamp':  datetime.now().isoformat(),
                            'prediction': int(pred),
                            'probability':float(proba),
                            'correct':    True,
                            'true_label': int(pred),
                            'data':       st.session_state.get('last_data', {})
                        }
                        save_feedback(entry)
                        st.session_state['fb_given'] = True
                        st.success("✅ Merci pour votre retour !")
                        st.rerun()

                with fb2:
                    if st.button("👎  Non, incorrecte", use_container_width=True):
                        st.session_state['fb_wrong'] = True

                if st.session_state.get('fb_wrong', False):
                    true_label = st.selectbox(
                        "Quelle était la vraie situation ?",
                        ["Le client est parti (Churn = 1)", "Le client est resté (Churn = 0)"]
                    )
                    if st.button("Confirmer la correction", use_container_width=True, type="primary"):
                        tl = 1 if "parti" in true_label else 0
                        entry = {
                            'timestamp':  datetime.now().isoformat(),
                            'prediction': int(pred),
                            'probability':float(proba),
                            'correct':    False,
                            'true_label': tl,
                            'data':       st.session_state.get('last_data', {})
                        }
                        save_feedback(entry)
                        st.session_state['fb_given'] = True
                        st.session_state['fb_wrong'] = False
                        st.success("✅ Correction enregistrée, merci !")
                        st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="padding:12px 16px; border-radius:10px;
                            background:#00d4aa11; border:1px solid #00d4aa44;
                            color:#00d4aa; font-size:0.85rem;">
                    ✅ Feedback enregistré — merci !
                </div>
                """, unsafe_allow_html=True)

        elif not model_ok:
            st.markdown("""
            <div class="card" style="text-align:center; padding:48px 24px;">
                <div style="font-size:3rem;">⚠️</div>
                <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; margin:12px 0 8px;">
                    Artifacts manquants
                </div>
                <div style="color:#64748b; font-size:0.85rem;">
                    Placez les fichiers <code>artifacts/</code> dans le répertoire racine du projet.
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE 2 — IMPORT CSV
# ─────────────────────────────────────────────
elif "CSV" in page:

    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-bottom:4px;">
        Prédiction en Lot
    </h1>
    <p style="color:#64748b; margin-bottom:32px;">
        Importez un fichier CSV avec les données clients pour obtenir des prédictions en masse.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📂 Import du fichier</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Glissez votre fichier CSV ici",
        type=['csv'],
        help="Colonnes requises : CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary"
    )
    st.markdown("""
    <p style="font-size:0.75rem; color:#64748b; margin-top:8px;">
        Colonnes requises : <code>CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary</code>
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded and model_ok:
        df_raw = pd.read_csv(uploaded)

        st.markdown('<div class="section-title">📊 Aperçu des données</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="metric-box">
                <div class="val">{len(df_raw):,}</div>
                <div class="lbl">Clients</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
                <div class="val">{df_raw.shape[1]}</div>
                <div class="lbl">Colonnes</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            missing = df_raw.isnull().sum().sum()
            st.markdown(f"""<div class="metric-box">
                <div class="val" style="color:{'#ff6b6b' if missing > 0 else '#00d4aa'};">{missing}</div>
                <div class="lbl">Valeurs manquantes</div>
            </div>""", unsafe_allow_html=True)

        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Lancer les prédictions", use_container_width=True, type="primary"):
            with st.spinner("Prédictions en cours..."):
                try:
                    preds, probas = predict_batch(df_raw, artifacts)

                    df_result = df_raw.copy()
                    df_result['Probabilité_Churn'] = np.round(probas, 4)
                    df_result['Prédiction']        = preds
                    df_result['Statut']            = df_result['Prédiction'].map({
                        1: '⚠️ Risque de départ',
                        0: '✅ Client fidèle'
                    })
                    df_result['Risque'] = pd.cut(
                        df_result['Probabilité_Churn'],
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['🟢 Faible', '🟡 Modéré', '🔴 Élevé']
                    )

                    st.session_state['batch_results'] = df_result
                    st.session_state['batch_preds']   = preds
                    st.session_state['batch_probas']  = probas

                except Exception as e:
                    st.error(f"Erreur : {e}")

        if 'batch_results' in st.session_state:
            df_result = st.session_state['batch_results']
            preds     = st.session_state['batch_preds']

            n_churn   = int(preds.sum())
            n_stay    = len(preds) - n_churn
            pct_churn = n_churn / len(preds) * 100

            st.markdown("---")
            st.markdown('<div class="section-title">🎯 Résultats</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-box">
                    <div class="val">{len(preds):,}</div>
                    <div class="lbl">Total analysés</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-box">
                    <div class="val" style="color:#ff6b6b;">{n_churn:,}</div>
                    <div class="lbl">Risque de départ</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-box">
                    <div class="val" style="color:#00d4aa;">{n_stay:,}</div>
                    <div class="lbl">Clients fidèles</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-box">
                    <div class="val" style="color:#f4c430;">{pct_churn:.1f}%</div>
                    <div class="lbl">Taux de churn</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(
                df_result[['Probabilité_Churn', 'Statut', 'Risque'] +
                          [c for c in df_result.columns
                           if c not in ['Probabilité_Churn', 'Statut', 'Risque', 'Prédiction']]],
                use_container_width=True
            )

            csv_export = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Télécharger les résultats (CSV)",
                data=csv_export,
                file_name=f"predictions_churn_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime='text/csv',
                use_container_width=True
            )

            # Feedback batch
            st.markdown("---")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">💬 Feedback sur les prédictions batch</div>', unsafe_allow_html=True)
            st.markdown('<p style="font-size:0.8rem; color:#64748b;">Si vous connaissez les vraies étiquettes, uploadez-les pour améliorer le modèle.</p>', unsafe_allow_html=True)

            true_labels_file = st.file_uploader(
                "Fichier CSV avec colonne 'Exited' (vraies étiquettes)",
                type=['csv'],
                key="true_labels"
            )
            if true_labels_file:
                df_true = pd.read_csv(true_labels_file)
                if 'Exited' in df_true.columns:
                    true_labels  = df_true['Exited'].values
                    preds_batch  = st.session_state['batch_preds']
                    probas_batch = st.session_state['batch_probas']

                    if len(true_labels) == len(preds_batch):
                        f2  = fbeta_score(true_labels, preds_batch, beta=2, zero_division=0)
                        acc = (true_labels == preds_batch).mean()

                        st.markdown(f"""
                        <div style="display:flex; gap:12px; margin:12px 0;">
                            <div class="metric-box" style="flex:1;">
                                <div class="val">{acc:.1%}</div>
                                <div class="lbl">Accuracy réelle</div>
                            </div>
                            <div class="metric-box" style="flex:1;">
                                <div class="val">{f2:.4f}</div>
                                <div class="lbl">F2-Score réel</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        if st.button("💾 Enregistrer ce feedback batch", use_container_width=True):
                            for p, prob, tl in zip(preds_batch, probas_batch, true_labels):
                                entry = {
                                    'timestamp':  datetime.now().isoformat(),
                                    'prediction': int(p),
                                    'probability':float(prob),
                                    'correct':    bool(p == tl),
                                    'true_label': int(tl),
                                    'source':     'batch'
                                }
                                save_feedback(entry)
                            st.success(f"✅ {len(preds_batch)} feedbacks enregistrés !")
                            st.rerun()
                    else:
                        st.error("⚠️ Le nombre de lignes ne correspond pas.")
                else:
                    st.error("⚠️ La colonne 'Exited' est introuvable dans le fichier.")
            st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PAGE 3 — DASHBOARD FEEDBACKS
# ─────────────────────────────────────────────
elif "Dashboard" in page:

    st.markdown("""
    <h1 style="font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; margin-bottom:4px;">
        Dashboard Feedbacks
    </h1>
    <p style="color:#64748b; margin-bottom:32px;">
        Suivi des retours utilisateurs et état du modèle.
    </p>
    """, unsafe_allow_html=True)

    feedbacks = load_feedbacks()
    needs_retrain, total_fb, wrong_fb = check_retrain_needed(20)

    if not feedbacks:
        st.markdown("""
        <div class="card" style="text-align:center; padding:48px;">
            <div style="font-size:3rem;">📭</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; margin:12px 0 8px;">
                Aucun feedback pour le moment
            </div>
            <div style="color:#64748b; font-size:0.85rem;">
                Les retours s'afficheront ici après les premières prédictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        correct_fb    = [f for f in feedbacks if f.get('correct') == True]
        wrong_fb_list = [f for f in feedbacks if f.get('correct') == False]
        accuracy_fb   = len(correct_fb) / len(feedbacks) * 100

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-box">
                <div class="val">{total_fb}</div>
                <div class="lbl">Total feedbacks</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-box">
                <div class="val" style="color:#00d4aa;">{len(correct_fb)}</div>
                <div class="lbl">Prédictions correctes</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-box">
                <div class="val" style="color:#ff6b6b;">{len(wrong_fb_list)}</div>
                <div class="lbl">Prédictions incorrectes</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-box">
                <div class="val" style="color:{'#00d4aa' if accuracy_fb >= 70 else '#ff6b6b'};">{accuracy_fb:.1f}%</div>
                <div class="lbl">Précision perçue</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-title">⚡ Progression vers le re-entraînement</div>', unsafe_allow_html=True)
        st.progress(min(total_fb / 20, 1.0))
        st.markdown(f"""
        <p style="font-size:0.8rem; color:#64748b; margin-top:4px;">
            {total_fb} / 20 feedbacks collectés
            {'— <span style="color:#f4c430; font-weight:600;">Re-entraînement disponible !</span>' if needs_retrain else ''}
        </p>
        """, unsafe_allow_html=True)

        if needs_retrain:
            if st.button("🔄 Re-entraîner maintenant", type="primary", use_container_width=True):
                with st.spinner("⏳ Re-entraînement en cours (~2-3 min)..."):
                    success, msg = retrain_model()
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.warning(msg)

        st.markdown("---")
        st.markdown('<div class="section-title">📋 Historique des feedbacks</div>', unsafe_allow_html=True)

        df_fb = pd.DataFrame(feedbacks)
        df_fb['correct_label']    = df_fb['correct'].map({True: '✅ Correct', False: '❌ Incorrect'})
        df_fb['prediction_label'] = df_fb['prediction'].map({1: '⚠️ Départ', 0: '✅ Fidèle'})
        df_fb['timestamp']        = pd.to_datetime(df_fb['timestamp']).dt.strftime('%d/%m/%Y %H:%M')

        cols_show = ['timestamp', 'prediction_label', 'probability', 'correct_label']
        if 'true_label' in df_fb.columns:
            df_fb['true_label_fmt'] = df_fb['true_label'].map({1: '⚠️ Départ', 0: '✅ Fidèle'}).fillna('—')
            cols_show.append('true_label_fmt')

        st.dataframe(
            df_fb[cols_show].rename(columns={
                'timestamp':       'Date',
                'prediction_label':'Prédiction',
                'probability':     'Probabilité',
                'correct_label':   'Feedback',
                'true_label_fmt':  'Vraie étiquette'
            }),
            use_container_width=True
        )

        csv_fb = pd.DataFrame(feedbacks).to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Exporter les feedbacks (CSV)",
            data=csv_fb,
            file_name=f"feedbacks_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
            use_container_width=True
        )