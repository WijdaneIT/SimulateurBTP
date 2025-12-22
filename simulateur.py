import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
import joblib
import time
import os
import shutil
import datetime
from sklearn.ensemble import GradientBoostingRegressor
from PIL import Image
import math
import gc
from pathlib import Path
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ---------------------------
# CONFIGURATION INITIALE
# ---------------------------
st.set_page_config(
    page_title="SimulIA BTP - Intelligence d'Achat",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration des chemins
DB_PATH = os.path.join(os.getcwd(), "simullA_btp.db") # Utilise le nom exact du fichier vu sur l'image
BACKUP_DIR = "backups/"
Path(BACKUP_DIR).mkdir(exist_ok=True)

# Configuration SMTP (à stocker dans les secrets Streamlit)
SMTP_CONFIG = {
    "server": st.secrets.get("SMTP_SERVER", "smtp.gmail.com"),
    "port": st.secrets.get("SMTP_PORT", 587),
    "username": st.secrets.get("SMTP_USERNAME", "simulia.btp@gmail.com"),
    "password": st.secrets.get("SMTP_PASSWORD", "")
}

# ---------------------------
# FONCTIONS UTILITAIRES
# ---------------------------
def backup_database():
    """Crée une sauvegarde horodatée de la base de données"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"simulIA_btp_{timestamp}.db")
    if os.path.exists(DB_PATH):
        shutil.copyfile(DB_PATH, backup_path)
    else:
        # créer un fichier vide si la DB n'existe pas encore
        open(backup_path, "wb").close()
    return backup_path

def show_splash_screen():
    """Affiche l'écran d'accueil"""
    splash = st.empty()
    with splash.container():
        st.title("SimulIA BTP")
        st.subheader("by Wijdane and Alaa")
        if os.path.exists("logo.png"):
            logo = Image.open("logo.png")
            st.image(logo, use_column_width=True)
        st.info("Chargement de l'intelligence d'Achat...")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.02)
            progress_bar.progress(i + 1)
    time.sleep(0.5)
    splash.empty()

def log_action(action, details=""):
    """Journalise les actions utilisateur"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("activity_log.txt", "a") as f:
        f.write(f"[{timestamp}] {action}: {details}\n")

def send_email(subject, body):
    """Envoie un email via SMTP"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_CONFIG["username"]
        msg['To'] = SMTP_CONFIG["username"]
        msg['Subject'] = subject

        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_CONFIG["server"], SMTP_CONFIG["port"])
        server.starttls()
        server.login(SMTP_CONFIG["username"], SMTP_CONFIG["password"])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        log_action("EMAIL_ERROR", str(e))
        return False

# ---------------------------
# BASE DE DONNÉES EMBARQUÉE
# ---------------------------
class BTPDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        # Important pour usage multi-threading de Streamlit
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()

        # Créer les tables si nécessaire (do this first)
        cursor.execute('''CREATE TABLE IF NOT EXISTS fournisseurs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            nom TEXT NOT NULL,
                            materiau TEXT NOT NULL,
                            prix_unitaire REAL,
                            qualite INTEGER CHECK(qualite BETWEEN 1 AND 10),
                            delai INTEGER,
                            fiabilite REAL CHECK(fiabilite BETWEEN 0 AND 1),
                            commandes_annuelles INTEGER DEFAULT 0,
                            durabilite REAL DEFAULT 5.0,
                            freq_entretien REAL DEFAULT 2.0,
                            recyclabilite REAL DEFAULT 0.8,
                            co2_unitaire REAL DEFAULT 0.159)''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS historique_decisions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            id_fournisseur INTEGER,
                            date_decision DATE DEFAULT CURRENT_DATE,
                            materiau TEXT,
                            budget REAL,
                            satisfaction INTEGER CHECK(satisfaction BETWEEN 1 AND 5),
                            cout_reel REAL,
                            delai_reel INTEGER,
                            FOREIGN KEY(id_fournisseur) REFERENCES fournisseurs(id))''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS ia_feedback (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            features TEXT,
                            prediction REAL,
                            resultat_reel REAL,
                            date_feedback DATE DEFAULT CURRENT_DATE)''')

        # Vérifier et ajouter les colonnes manquantes (après création)
        cursor.execute("PRAGMA table_info(fournisseurs)")
        columns = [col[1] for col in cursor.fetchall()]

        new_columns = [
            ('durabilite', 'REAL', 5.0),
            ('freq_entretien', 'REAL', 2.0),
            ('recyclabilite', 'REAL', 0.8),
            ('co2_unitaire', 'REAL', 0.159)
        ]

        for col_name, col_type, default_val in new_columns:
            if col_name not in columns:
                cursor.execute(f"ALTER TABLE fournisseurs ADD COLUMN {col_name} {col_type} DEFAULT {default_val}")

        self.conn.commit()

    @st.cache_data(ttl=300)
    def get_fournisseurs(_self):
        return pd.read_sql_query("SELECT * FROM fournisseurs", _self.conn)

    def add_fournisseur(self, fournisseur):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO fournisseurs 
                          (nom, materiau, prix_unitaire, qualite, delai, fiabilite, 
                           durabilite, freq_entretien, recyclabilite, co2_unitaire) 
                          VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                       (fournisseur['nom'], fournisseur['materiau'],
                        fournisseur['prix_unitaire'], fournisseur['qualite'],
                        fournisseur['delai'], fournisseur['fiabilite'],
                        fournisseur.get('durabilite', 5.0),
                        fournisseur.get('freq_entretien', 2.0),
                        fournisseur.get('recyclabilite', 0.8),
                        fournisseur.get('co2_unitaire', 0.159)))
        self.conn.commit()
        log_action("AJOUT_FOURNISSEUR", f"{fournisseur['nom']} - {fournisseur['materiau']}")

    def record_decision(self, decision):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO historique_decisions 
                          (id_fournisseur, materiau, budget, satisfaction, cout_reel, delai_reel) 
                          VALUES (?, ?, ?, ?, ?, ?)''',
                       (decision['id_fournisseur'], decision['materiau'],
                        decision['budget'], decision['satisfaction'],
                        decision['cout_reel'], decision['delai_reel']))
        self.conn.commit()
        log_action("DECISION", f"Fournisseur: {decision['id_fournisseur']} - Satisfaction: {decision['satisfaction']}")

    def close(self):
        self.conn.close()

# ---------------------------
# MODÈLES IA EMBARQUÉS
# ---------------------------
class IAEngine:
    def __init__(self, db):
        self.db = db
        self.load_models()

    def load_models(self):
        # On initialise d'abord des modèles par défaut
        self.cout_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        self.delai_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        
        # On essaie de charger les modèles sauvegardés
        if os.path.exists("model_cout_cache.pkl"):
            try:
                self.cout_model = joblib.load("model_cout_cache.pkl")
            except: pass # On garde le modèle par défaut si le chargement échoue
        
        if os.path.exists("model_delai.pkl"):
            try:
                self.delai_model = joblib.load("model_delai.pkl")
            except: pass

    def save_models(self):
        joblib.dump(self.cout_model, "model_cout_cache.pkl")
        joblib.dump(self.delai_model, "model_delai.pkl")

    # Ne pas cache l'entraînement (il modifie l'état)
    def train_models(self):
        df = pd.read_sql_query('''SELECT f.prix_unitaire, f.qualite, f.delai, f.fiabilite, 
                                         h.cout_reel, h.delai_reel 
                                  FROM historique_decisions h
                                  JOIN fournisseurs f ON h.id_fournisseur = f.id 
                                  WHERE h.cout_reel IS NOT NULL AND h.delai_reel IS NOT NULL''', self.db.conn)
        if len(df) > 10:
            X = df[['prix_unitaire', 'qualite', 'delai', 'fiabilite']]
            y_cout = df['cout_reel'] - df['prix_unitaire']
            y_delai = df['delai_reel']
            self.cout_model.fit(X, y_cout)
            self.delai_model.fit(X, y_delai)
            self.save_models()

    def predict_cout_cache(self, features):
        try:
            return self.cout_model.predict([features])[0]
        except:
            prix, qualite, delai, fiabilite = features
            cout_cache = 0
            if qualite < 5:
                cout_cache += prix * 0.03
            if delai > 7:
                cout_cache += prix * 0.02
            return cout_cache

    def predict_delai_reel(self, features):
        try:
            return self.delai_model.predict([features])[0]
        except:
            return features[2]

# ---------------------------
# FONCTIONS D'INTERFACE
# ---------------------------
def setup_theme():
    """Configure le thème sombre/clair"""
    st.sidebar.header("Préférences")
    theme = st.sidebar.radio("Thème", ["Clair", "Sombre"], index=0)
    if theme == "Sombre":
        st.markdown("""
        <style>
            .stApp { background-color: #1e1e1e; color: #ffffff; }
            .css-18e3th9 { background-color: #2e2e2e; }
            .css-1d391kg { background-color: #2e2e2e; }
        </style>
        """, unsafe_allow_html=True)

def show_database_management(db):
    """Affiche les outils de gestion de la base"""
    with st.sidebar.expander("🔧 Gestion de la base"):
        st.write("**Sauvegardes**")
        if st.button("💾 Créer une sauvegarde"):
            backup_path = backup_database()
            st.success(f"Sauvegarde créée: {backup_path}")
            log_action("SAUVEGARDE", backup_path)

        st.write("**Importer une base**")
        uploaded_db = st.file_uploader("Choisir un fichier .db", type="db")
        if uploaded_db is not None:
            if st.button("⚠️ Confirmer la substitution de la base actuelle"):
                with open(DB_PATH, "wb") as f:
                    f.write(uploaded_db.getvalue())
                st.success("Base de données importée avec succès!")
                log_action("IMPORT_DB", uploaded_db.name)
                st.experimental_rerun()

        st.write("**Exporter les données**")

        if st.button("📤 Exporter les fournisseurs (CSV)"):
            df = db.get_fournisseurs()
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger CSV",
                data=csv,
                file_name="fournisseurs.csv",
                mime="text/csv"
            )


def show_help_contacts():
    """Affiche les options d'aide et contact"""
    with st.sidebar.expander("❓ Aide & Contact"):
        st.write("**Documentation**")
        if st.button("📚 Voir le tutoriel"):
            show_tutorial()

        st.write("**Support**")
        bug_description = st.text_area("Décrire le problème", key="bug_desc")
        if st.button("✉️ Envoyer le rapport"):
            if bug_description:
                # Enregistrement local
                log_action("BUG_REPORT", bug_description)

                # Envoi par email
                subject = f"[SimulIA BTP] Rapport de bug - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                body = f"Un nouveau bug a été signalé :\n\n{bug_description}"

                if send_email(subject, body):
                    st.success("Merci! Votre rapport a été envoyé à notre équipe.")
                else:
                    st.warning("Votre rapport a été enregistré localement mais n'a pas pu être envoyé par email.")
            else:
                st.warning("Veuillez décrire le problème avant d'envoyer le rapport")

        st.write("**Contact**")
        st.markdown("[✉️ Nous contacter](mailto:simulia.btp@gmail.com)")

def show_tutorial():
    """Affiche le tutoriel d'utilisation"""
    with st.expander("🎓 Tutoriel - Premiers pas", expanded=True):
        tabs = st.tabs(["1. Import initial", "2. Ajouter fournisseurs", "3. Simuler un achat"])
        with tabs[0]:
            st.write("""
            **Étape 1: Import initial**  
            - Utilisez l'outil d'import dans 'Gestion de la base'  
            - Chargez un fichier .db existant ou commencez avec une base vide
            """)
        with tabs[1]:
            st.write("""
            **Étape 2: Ajouter des fournisseurs**  
            - Allez dans l'onglet 'Configuration'  
            - Remplissez le formulaire 'Ajouter un fournisseur'  
            - Complétez les critères avancés si nécessaire
            """)
        with tabs[2]:
            st.write("""
            **Étape 3: Simuler un achat**  
            - Naviguez vers l'onglet 'Simulation'  
            - Sélectionnez votre matériau et critères  
            - Analysez les recommandations de l'IA
            """)
        st.info("Utilisez les outils de gestion en bas à gauche pour exporter vos données ou créer des sauvegardes")

# ---------------------------
# FONCTIONS DE CALCUL
# ---------------------------
@st.cache_data
def appliquer_surcharges_ia(df, _ia_engine):  # Correction du paramètre ici
    df = df.copy()
    for col in ['durabilite', 'freq_entretien', 'recyclabilite', 'co2_unitaire']:
        if col not in df.columns:
            df[col] = {
                'durabilite': 5.0,
                'freq_entretien': 2.0,
                'recyclabilite': 0.8,
                'co2_unitaire': 0.159
            }[col]

    df["cout_cache"] = 0.0
    df["delai_estime"] = df["delai"]
    df["tco"] = df["prix_unitaire"] + (df["durabilite"] / df["freq_entretien"]) * 0.1 * df["prix_unitaire"]

    for idx, row in df.iterrows():
        features = [row['prix_unitaire'], row['qualite'], row['delai'], row['fiabilite']]
        df.at[idx, "cout_cache"] = _ia_engine.predict_cout_cache(features)  # Utilisation correcte
        df.at[idx, "delai_estime"] = _ia_engine.predict_delai_reel(features)  # Utilisation correcte

    df["prix_total"] = df["prix_unitaire"] + df["cout_cache"]
    return df

@st.cache_data
def normaliser(_df, colonne):
    if colonne not in _df.columns or _df[colonne].nunique() == 0:
        return pd.Series([0.5]*len(_df))
    min_val, max_val = _df[colonne].min(), _df[colonne].max()
    if max_val - min_val == 0:
        return pd.Series([0.5]*len(_df))
    return (_df[colonne] - min_val) / (max_val - min_val)

@st.cache_data
def calculer_scores(_df, poids):
    df = _df.copy()
    df["score_prix"] = 1 - normaliser(df, "prix_total")
    df["score_qualite"] = df["qualite"] / 10
    df["score_delai"] = 1 - normaliser(df, "delai_estime")
    df["score_fiabilite"] = df["fiabilite"]
    df["score_dependance"] = 1 - normaliser(df, "commandes_annuelles")
    df["score_ecologie"] = (df["recyclabilite"] + (1 - normaliser(df, "co2_unitaire"))) / 2
    df["score_tco"] = 1 - normaliser(df, "tco")

    total_poids = sum(poids.values()) if sum(poids.values()) > 0 else 1.0
    poids_normalises = {k: v / total_poids for k, v in poids.items()}

    df["score_total"] = (
        df["score_prix"] * poids_normalises.get("prix", 0) +
        df["score_qualite"] * poids_normalises.get("qualite", 0) +
        df["score_delai"] * poids_normalises.get("delai", 0) +
        df["score_fiabilite"] * poids_normalises.get("fiabilite", 0) +
        df["score_dependance"] * poids_normalises.get("dependance", 0) +
        df["score_ecologie"] * poids_normalises.get("ecologie", 0) +
        df["score_tco"] * poids_normalises.get("tco", 0)
    )
    return df

# ---------------------------
# INTERFACE PRINCIPALE
# ---------------------------
def configuration_tab(db, df):
    """Onglet de configuration des fournisseurs"""
    st.header("🔧 Configuration des fournisseurs")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Ajouter un fournisseur")
        with st.form(key='form_fournisseur'):
            nom = st.text_input("Nom du fournisseur*", help="Nom commercial de l'entreprise")
            materiau_fournisseur = st.text_input("Matériau fourni*", help="Type de matériau (ciment, bois, etc.)")
            prix = st.number_input("Prix unitaire (€)*", min_value=0.0, value=0.0, step=0.1)
            qualite = st.slider("Qualité (1-10)*", 1, 10, 5, help="Note subjective de la qualité des produits")
            delai = st.number_input("Délai moyen (jours)*", min_value=0, value=0)
            fiabilite = st.slider("Taux de fiabilité (0-1)*", 0.0, 1.0, 1.0, step=0.1,
                                 help="Probabilité de respect des engagements")

            with st.expander("⚙️ Critères avancés (optionnel)"):
                durabilite = st.number_input("Durée de vie (années)", min_value=0.0, value=5.0, step=0.5)
                freq_entretien = st.number_input("Fréquence entretien (années)", min_value=0.0, value=2.0, step=0.5)
                recyclabilite = st.slider("Taux de recyclabilité (%)", 0, 100, 80)
                co2_unitaire = st.number_input("Émission CO₂ (kg/unité)", min_value=0.0, value=0.159, step=0.01)

            submitted = st.form_submit_button("✅ Ajouter ce fournisseur")

        if submitted:
            if not nom or not materiau_fournisseur:
                st.error("Les champs obligatoires (*) doivent être remplis")
            else:
                fournisseur = {
                    'nom': nom,
                    'materiau': materiau_fournisseur,
                    'prix_unitaire': prix,
                    'qualite': qualite,
                    'delai': delai,
                    'fiabilite': fiabilite,
                    'durabilite': durabilite,
                    'freq_entretien': freq_entretien,
                    'recyclabilite': recyclabilite / 100,
                    'co2_unitaire': co2_unitaire
                }
                db.add_fournisseur(fournisseur)
                st.success(f"Fournisseur {nom} ajouté avec succès!")
                st.experimental_rerun()

    with col2:
        st.subheader("Liste des fournisseurs")
        if not df.empty:
            search_term = st.text_input("🔍 Rechercher par nom")
            if search_term:
                filtered_df = df[df['nom'].str.contains(search_term, case=False)]
            else:
                filtered_df = df

            st.dataframe(filtered_df[['nom', 'materiau', 'prix_unitaire', 'qualite']],
                         height=500, use_container_width=True)
        else:
            st.info("Aucun fournisseur enregistré")

def simulation_tab(db, ia_engine, df):  # Correction du paramètre ici
    """Onglet de simulation d'achat"""
    st.header("📊 Simulation d'achat")

    # Section Paramètres
    with st.expander("⚙️ Paramètres de simulation", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Mon achat")
            materiau = st.selectbox("Matériau requis*", df["materiau"].unique())
            default_budget = float(df[df["materiau"]==materiau]["prix_unitaire"].max() * 1.2) if not df[df["materiau"]==materiau].empty else 100.0
            budget_max = st.number_input("Budget maximum (€)*", min_value=0.0,
                                       value=default_budget,
                                       step=100.0)
            delai_max = st.number_input("Délai maximum (jours)*", min_value=0, value=10)

        with col2:
            st.subheader("Critères de décision")
            poids = {
                "prix": st.slider("Importance du prix*", 0.0, 1.0, 0.3),
                "qualite": st.slider("Importance de la qualité*", 0.0, 1.0, 0.25),
                "delai": st.slider("Importance du délai*", 0.0, 1.0, 0.2),
                "fiabilite": st.slider("Importance de la fiabilité*", 0.0, 1.0, 0.15),
                "dependance": st.slider("Réduction dépendance fournisseur*", 0.0, 0.5, 0.1)
            }

            with st.expander("♻️ Critères avancés"):
                poids["ecologie"] = st.slider("Impact écologique", 0.0, 1.0, 0.1)
                poids["tco"] = st.slider("Coût total de possession", 0.0, 1.0, 0.1)

    # Traitement des données
    start_time = time.time()
    df_ia = appliquer_surcharges_ia(df, ia_engine)  # Utilisation correcte
    df_scores = calculer_scores(df_ia, poids)
    df_filtre = df_scores[(df_scores["materiau"] == materiau) &
                         (df_scores["prix_total"] <= budget_max) &
                         (df_scores["delai_estime"] <= delai_max)]

    # Affichage des résultats
    if df_filtre.empty:
        st.warning("⚠️ Aucun fournisseur ne répond à vos critères")
        st.info("Conseil IA: Augmentez votre budget de 15% ou le délai de 5 jours")
        return

    df_tries = df_filtre.sort_values("score_total", ascending=False)
    meilleur = df_tries.iloc[0]

    st.subheader("🔍 Recommandation d'achat")
    col1, col2, col3 = st.columns(3)
    col1.metric("Fournisseur recommandé", meilleur["nom"])
    col2.metric("Prix total estimé", f"{meilleur['prix_total']:.2f}€")
    risk_pct = (meilleur['cout_cache']/meilleur['prix_unitaire']*100) if meilleur['prix_unitaire'] else 0.0
    col3.metric("Risque total", f"{risk_pct:.1f}%")

    # Tableau comparatif
    st.subheader("📋 Comparaison des fournisseurs")
    colonnes_principales = ['nom', 'prix_unitaire', 'cout_cache', 'prix_total',
                           'qualite', 'delai_estime', 'fiabilite', 'score_total']
    st.dataframe(df_tries[colonnes_principales].sort_values('score_total', ascending=False),
                height=400, use_container_width=True)

    # Détails avancés
    with st.expander("📈 Analyse détaillée"):
        st.subheader("Performance environnementale")
        col4, col5, col6 = st.columns(3)
        col4.metric("Impact CO₂", f"{meilleur['co2_unitaire']} kg/unité")
        col5.metric("Recyclabilité", f"{meilleur['recyclabilite']*100:.0f}%")
        col6.metric("Coût total (TCO)", f"{meilleur['tco']:.2f}€")

        st.subheader("Analyse comparative complète")
        colonnes_avancees = ['durabilite', 'freq_entretien', 'recyclabilite',
                            'co2_unitaire', 'tco', 'score_ecologie', 'score_tco']
        st.dataframe(df_tries[colonnes_principales + colonnes_avancees].sort_values('score_total', ascending=False))

    # Feedback
    with st.expander("💾 Enregistrer cette décision"):
        satisfaction = st.slider("Satisfaction (1-5)", 1, 5, 3,
                                help="Votre estimation de satisfaction pour cette décision")
        cout_reel = st.number_input("Coût réel (€)", value=float(meilleur['prix_total']))
        delai_reel = st.number_input("Délai réel (jours)", value=int(meilleur['delai_estime']))

        if st.button("Sauvegarder la décision"):
            decision = {
                'id_fournisseur': int(meilleur['id']),
                'materiau': materiau,
                'budget': budget_max,
                'satisfaction': satisfaction,
                'cout_reel': cout_reel,
                'delai_reel': delai_reel
            }
            db.record_decision(decision)
            st.success("Décision enregistrée dans l'historique!")

    # Performance
    st.caption(f"Temps de traitement: {time.time() - start_time:.2f} secondes")

# ---------------------------
# APPLICATION PRINCIPALE
# ---------------------------
def main():
    # Backup initial
    backup_path = backup_database()
    log_action("APP_START", f"Backup créé: {backup_path}")

    # Interface
    show_splash_screen()
    setup_theme()

    # Initialisation base de données
    db = BTPDatabase()
    ia_engine = IAEngine(db)

    # Chargement données avec cache
    with st.spinner("Chargement des données..."):
        df = db.get_fournisseurs()

    # Entraînement IA
    with st.spinner("Optimisation de l'intelligence d'Achat..."):
        ia_engine.train_models()

    # Sidebar (passer db)
    show_database_management(db)
    show_help_contacts()

    # Tutoriel
    show_tutorial()

    # Onglets principaux
    tab1, tab2 = st.tabs(["⚙️ Configuration", "📊 Simulation"])

    with tab1:
        configuration_tab(db, df)

    with tab2:
        if not df.empty:
            simulation_tab(db, ia_engine, df)  # Correction ici
        else:
            st.warning("Ajoutez d'abord des fournisseurs dans l'onglet Configuration")

    # Nettoyage mémoire
    gc.collect()

if __name__ == "__main__":
    main()
