"""Streamlit web app: Algérie Télécom tableau de prévision des encaissements."""
import requests
import pandas as pd
import streamlit as st

BACKEND = "http://127.0.0.1:8000"

st.set_page_config(page_title="Algérie Télécom — Prévision des encaissements", layout="wide")

st.markdown("""
<style>
.hero { background: linear-gradient(90deg, #0a5c36, #0ea5a0); padding: 28px 32px;
        border-radius: 14px; color: white; margin-bottom: 18px; }
.hero h1 { margin: 0; font-size: 30px; }
.hero p { margin: 6px 0 0 0; opacity: 0.92; }
.card { background: #f7faf9; border: 1px solid #e5eee9; border-radius: 12px; padding: 14px 18px; }
.small { color: #5b6b62; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>Algérie Télécom — Prévision des encaissements</h1>
<p>Comportement de paiement espèces vs en ligne : ce que les clients ont payé hier, ce que le modèle prévoit pour demain.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Ce que fait cette application :** elle apprend à partir des encaissements quotidiens (espèces payées aux guichets, paiements en ligne)
et prédit les prochains jours en **montants encaissés (DZD)** et **nombre de transactions**.
Utilisez la prévision pour anticiper la trésorerie, le personnel aux points de paiement et la bascule vers le paiement en ligne.
""")

# --- Sidebar: settings ---
st.sidebar.header("Paramètres")
days = st.sidebar.slider("Jours futurs à prédire", 7, 90, 30)
st.sidebar.markdown("---")
st.sidebar.markdown("**API Backend**")
st.sidebar.code(BACKEND, language="text")
st.sidebar.markdown("<span class='small'>Le backend doit être en cours d'exécution :<br/>"
                    "<code>uvicorn backend:app --app-dir app</code></span>", unsafe_allow_html=True)

# --- Load data ---
try:
    info = requests.get(f"{BACKEND}/", timeout=30).json()
    hist = pd.DataFrame(requests.get(f"{BACKEND}/history", timeout=30).json())
    hist["date"] = pd.to_datetime(hist["date"])
    pred = requests.get(f"{BACKEND}/predict", params={"days": days}, timeout=30).json()
    fut = pd.DataFrame({
        "date": pd.to_datetime(pred["dates"]),
        "cash_amount": pred["cash_amount"],
        "online_amount": pred["online_amount"],
        "cash_count": pred["cash_count"],
        "online_count": pred["online_count"],
    })
except Exception as e:
    st.error(f"Backend inaccessible à {BACKEND}. Démarrez-le d'abord, puis actualisez. Erreur : {e}")
    st.stop()

# --- KPI row ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Jours d'historique", f"{len(hist)}",
          f"{hist['date'].min().date()} → {hist['date'].max().date()}")
c2.metric("Espèces encaissées (historique)", f"{hist['cash_amount'].sum()/1e6:.1f}M DZD",
          f"{int(hist['cash_count'].sum()):,} transactions")
c3.metric(f"Prévision espèces (prochains {days} j)", f"{fut['cash_amount'].sum()/1e6:.1f}M DZD",
          f"{int(fut['cash_count'].sum()):,} transactions")
c4.metric(f"Prévision en ligne (prochains {days} j)", f"{fut['online_amount'].sum()/1e6:.1f}M DZD",
          f"{int(fut['online_count'].sum()):,} transactions")

st.markdown("---")
st.header("Prévisions dans le temps")
st.markdown("Chaque graphique montre **l'historique (passé)** et **la prévision (futur)**. "
            "La prévision est récursive : chaque jour prédit devient l'entrée du jour suivant.")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("Montant espèces (DZD par jour)")
    st.markdown("<span class='small'>Argent encaissé en espèces aux guichets.</span>", unsafe_allow_html=True)
    c = pd.concat([hist[["date", "cash_amount"]].assign(kind="historique"),
                   fut[["date", "cash_amount"]].assign(kind="prévision")])
    st.line_chart(c.pivot(index="date", columns="kind", values="cash_amount"))
with col_b:
    st.subheader("Montant en ligne (DZD par jour)")
    st.markdown("<span class='small'>Argent payé en ligne. En croissance depuis juin — à surveiller.</span>",
                unsafe_allow_html=True)
    o = pd.concat([hist[["date", "online_amount"]].assign(kind="historique"),
                   fut[["date", "online_amount"]].assign(kind="prévision")])
    st.line_chart(o.pivot(index="date", columns="kind", values="online_amount"))

col_c, col_d = st.columns(2)
with col_c:
    st.subheader("Comparaison : prévision espèces vs en ligne (montant)")
    st.line_chart(fut.set_index("date")[["cash_amount", "online_amount"]])
with col_d:
    st.subheader("Comparaison : prévision espèces vs en ligne (transactions)")
    st.line_chart(fut.set_index("date")[["cash_count", "online_count"]])

with st.expander(f"Tableau des prévisions — prochains {days} jours (cliquer pour ouvrir)"):
    st.dataframe(fut, use_container_width=True)
    st.download_button("Télécharger la prévision en CSV", fut.to_csv(index=False), "forecast.csv", "text/csv")

st.markdown("---")
st.header("Mettre à jour le modèle avec des données récentes")
st.markdown("""
Téléversez un **fichier de paiement plus récent** (même format Excel : feuilles mensuelles, 12 colonnes).
Choisissez une option — le backend réentraîne le modèle et l'enregistre, la nouvelle prévision s'applique immédiatement.
""")

up1, up2 = st.columns(2)
with up1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Ajouter — agrandir le jeu de données")
    st.markdown("Ajoute le nouveau fichier à l'historique existant. "
                "Sur les dates en commun, les valeurs sont **additionnées**. Idéal quand le nouveau fichier contient des **mois plus récents**.")
    f_append = st.file_uploader("Fichier Excel récent à ajouter", type=["xlsx", "xls"], key="append")
    if st.button("Ajouter + Réentraîner", type="primary", use_container_width=True):
        if f_append is None:
            st.warning("Veuillez d'abord choisir un fichier Excel.")
        else:
            with st.spinner("Ajout des données et réentraînement… (1–2 min)"):
                try:
                    r = requests.post(f"{BACKEND}/retrain/append",
                                      files={"file": (f_append.name, f_append.getvalue())}, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Échec du réentraînement : {r.text}")
                    else:
                        res = r.json()
                        st.success(f"Terminé ! Désormais {res['rows']} jours "
                                   f"({res['date_from']} → {res['date_to']}). Gagnants : {res['best_models']}")
                        st.json(res)
                        st.info("Actualisez la page pour voir les prévisions issues du jeu de données agrandi.")
                except Exception as e:
                    st.error(f"Échec du téléversement : {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with up2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Remplacer — repartir de zéro")
    st.markdown("Supprime l'ancien historique et entraîne **uniquement sur le nouveau fichier**. "
                "À utiliser quand le nouveau fichier est une **version corrigée ou complète**.")
    f_replace = st.file_uploader("Fichier Excel récent pour remplacer", type=["xlsx", "xls"], key="replace")
    if st.button("Remplacer + Réentraîner", use_container_width=True):
        if f_replace is None:
            st.warning("Veuillez d'abord choisir un fichier Excel.")
        else:
            with st.spinner("Remplacement des données et réentraînement… (1–2 min)"):
                try:
                    r = requests.post(f"{BACKEND}/retrain/replace",
                                      files={"file": (f_replace.name, f_replace.getvalue())}, timeout=300)
                    if r.status_code != 200:
                        st.error(f"Échec du réentraînement : {r.text}")
                    else:
                        res = r.json()
                        st.success(f"Terminé ! Désormais {res['rows']} jours "
                                   f"({res['date_from']} → {res['date_to']}). Gagnants : {res['best_models']}")
                        st.json(res)
                        st.info("Actualisez la page pour voir les prévisions issues du nouveau jeu de données.")
                except Exception as e:
                    st.error(f"Échec du téléversement : {e}")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.header("À propos du modèle")
st.markdown(f"""
- **Cibles prédites :** montant espèces, montant en ligne, nombre espèces, nombre en ligne (un modèle chacun).
- **Gagnants actuels (MAE la plus faible) :** `{info.get('best_models')}`.
- **Variables par jour :** tendance, jour de la semaine, mois, jour du mois, indicateur week-end, veille (`lag_1`),
  même jour la semaine précédente (`lag_7`), moyenne des 7 derniers jours (`roll_7`).
- **Entraînement :** 4 modèles comparés par cible (LinearRegression, Ridge, RandomForest, GradientBoosting)
  sur un découpage chronologique 80/20 ; gagnants réentraînés sur tout l'historique. Voir `TECHNICAL.md` pour les détails.
""")

st.markdown("<p class='small'>Algérie Télécom — tableau de bord interne de prévision. "
            "Les prévisions sont des estimations basées sur les tendances passées, sans garantie.</p>", unsafe_allow_html=True)
