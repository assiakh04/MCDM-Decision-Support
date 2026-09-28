import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mcda.methods import (
    ahp_weights, bwm_weights, entropy_weights, critic_weights,
    equal_weights, wsm, wpm, waspas, topsis, vikor,
    compare_rankings, spearman_matrix
)

st.set_page_config(
    page_title="MCDM Studio — Centre logistique",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("assets/style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

PASTELS = ["#7C83FD", "#F38BA0", "#58C4A3", "#FDBA74", "#7AB8FF",
           "#B28DFF", "#8BD3C7", "#FF9EB5", "#91B8F6", "#F7B5A7"]
DIRECTIONS = ["Bénéfice (+)", "Coût (-)"]
PAGES = [
    "1. Scénario & données",
    "2. Pondération",
    "3. Classement",
    "4. Comparaison",
    "5. Sensibilité",
    "6. Guide des méthodes"
]

# ---------------------------------------------------------------------
# DATA & STATE
# ---------------------------------------------------------------------
def demo_data():
    df = pd.DataFrame({
        "Coût terrain (M DH)": [12.5, 10.8, 8.7, 11.6, 7.9, 9.4, 8.2, 6.9, 6.5, 5.8],
        "Accessibilité (/10)": [9.5, 9.2, 8.8, 9.0, 8.1, 7.8, 7.2, 7.4, 7.6, 6.8],
        "Port/Aéroport (/10)": [9.8, 9.6, 7.6, 8.7, 8.4, 8.0, 8.6, 7.0, 6.8, 7.4],
        "Distance clients (km)": [18, 32, 29, 22, 37, 41, 58, 52, 49, 63],
        "Expansion (/10)": [7.0, 8.4, 9.1, 6.8, 8.7, 8.0, 8.3, 8.8, 9.0, 9.2],
        "Main-d'œuvre (/10)": [9.4, 8.2, 8.6, 9.0, 7.4, 8.1, 7.9, 7.6, 7.7, 7.0],
        "Impact env. (/10)": [7.1, 6.0, 4.8, 6.5, 4.4, 5.6, 4.1, 3.8, 3.5, 3.2],
        "Infrastructures (/10)": [9.6, 9.0, 8.7, 9.1, 8.2, 8.4, 8.0, 7.9, 8.1, 7.5],
    }, index=[
        "Casablanca", "Tanger", "Kénitra", "Rabat", "El Jadida",
        "Marrakech", "Agadir", "Fès", "Meknès", "Oujda"
    ])

    meta = pd.DataFrame({
        "Critère": df.columns,
        "Sens": [
            "Coût (-)", "Bénéfice (+)", "Bénéfice (+)", "Coût (-)",
            "Bénéfice (+)", "Bénéfice (+)", "Coût (-)", "Bénéfice (+)"
        ]
    })
    return df, meta


def init_state():
    if "df" not in st.session_state or "meta" not in st.session_state:
        df, meta = demo_data()
        st.session_state.df = df
        st.session_state.meta = meta
    if "weights" not in st.session_state:
        st.session_state.weights = None
    if "weight_method" not in st.session_state:
        st.session_state.weight_method = "Non calculée"


init_state()


def current_directions():
    meta = st.session_state.meta.copy()
    meta = meta[meta["Critère"].isin(st.session_state.df.columns)]
    return dict(zip(meta["Critère"], meta["Sens"]))


# ---------------------------------------------------------------------
# HELPERS UI
# ---------------------------------------------------------------------
def page_badge(text, tone="neutral"):
    st.markdown(f'<span class="soft-badge {tone}">{text}</span>', unsafe_allow_html=True)


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title, text, icon="✦"):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-icon">{icon}</div>
            <div>
                <div class="info-card-title">{title}</div>
                <div class="info-card-text">{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(
        f"""
        <div class="section-header">
            <div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(page_name):
    weight_method = st.session_state.get("weight_method", "Non calculée")
    st.markdown(
        f"""
        <div class="hero-app">
            <div class="hero-content">
                <div class="hero-kicker">MCDM Studio • application pédagogique</div>
                <h1>MCDM Studio</h1>
                <p>
                    Système d'aide à la décision multicritère avec une interface pensée comme une vraie application :
                    élégante, lisible, et prête pour une démonstration orale.
                </p>
                <div class="hero-badges">
                    <span class="soft-badge lavender">Scénario : Centre logistique</span>
                    <span class="soft-badge mint">Page active : {page_name}</span>
                    <span class="soft-badge peach">Pondération : {weight_method}</span>
                </div>
            </div>
            <div class="hero-aside">
                <div class="mini-card">
                    <div class="mini-top">Objectif</div>
                    <div class="mini-main">Comparer, pondérer, classer et expliquer la meilleure alternative.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_ribbon():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Alternatives", len(st.session_state.df), "modifiables")
    with c2:
        kpi_card("Critères", len(st.session_state.df.columns), "+ et -")
    with c3:
        computed = "Oui" if st.session_state.get("weights") is not None else "Non"
        kpi_card("Poids calculés", computed, st.session_state.get("weight_method", "—"))
    with c4:
        positives = int((st.session_state.meta["Sens"] == "Bénéfice (+)").sum())
        negatives = int((st.session_state.meta["Sens"] == "Coût (-)").sum())
        kpi_card("Mix des critères", f"{positives} / {negatives}", "bénéfices / coûts")


def pastel_bar(series, title, higher_better=True, horizontal=True):
    data = pd.DataFrame({"Alternative": series.index, "Valeur": series.values})
    data = data.sort_values("Valeur", ascending=not higher_better)
    if horizontal:
        fig = px.bar(data, x="Valeur", y="Alternative", orientation="h", title=title)
    else:
        fig = px.bar(data, x="Alternative", y="Valeur", title=title)
    fig.update_traces(
        marker_color=[PASTELS[i % len(PASTELS)] for i in range(len(data))],
        marker_line_color="#ffffff",
        marker_line_width=1.5,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=20),
        title_x=0.02,
        height=max(380, 36 * len(data)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
    )
    return fig


def weight_chart(weights):
    fig = px.bar(
        x=weights.index, y=weights.values,
        labels={"x": "Critère", "y": "Poids"},
        title="Poids des critères"
    )
    fig.update_traces(marker_color=[PASTELS[i % len(PASTELS)] for i in range(len(weights))])
    fig.update_layout(template="plotly_white", title_x=0.02, height=390, paper_bgcolor="rgba(0,0,0,0)")
    return fig


def rank_table(scores, higher=True):
    d = pd.DataFrame({"Score": scores})
    d["Rang"] = d["Score"].rank(ascending=not higher, method="min").astype(int)
    return d.sort_values(["Rang", "Score"], ascending=[True, not higher])


def podium_cards(result_df):
    if result_df.empty:
        return
    ordered = result_df.sort_values("Rang").head(3)
    emojis = ["🥇", "🥈", "🥉"]
    cols = st.columns(3)
    for idx, (name, row) in enumerate(ordered.iterrows()):
        with cols[idx]:
            st.markdown(
                f"""
                <div class="podium-card podium-{idx+1}">
                    <div class="podium-rank">{emojis[idx]} Rang {int(row['Rang'])}</div>
                    <div class="podium-name">{name}</div>
                    <div class="podium-score">Score : {row['Score']:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def make_overview_figures(df, directions):
    df2 = df.copy()
    size_col = next((c for c in df.columns if "Infrastructures" in c), df.columns[-1])
    x_col = next((c for c in df.columns if "Coût" in c), df.columns[0])
    y_col = next((c for c in df.columns if "Distance" in c), df.columns[min(1, len(df.columns)-1)])
    color_values = ["Bénéfice" if directions.get(c) == "Bénéfice (+)" else "Coût" for c in df.columns]

    fig_scatter = px.scatter(
        df2.reset_index().rename(columns={"index": "Alternative"}),
        x=x_col, y=y_col, size=size_col, color="Alternative", text="Alternative",
        title="Lecture rapide du scénario"
    )
    fig_scatter.update_traces(textposition="top center")
    fig_scatter.update_layout(template="plotly_white", title_x=0.02, height=430, showlegend=False)

    counts = pd.Series(color_values).value_counts()
    fig_donut = px.pie(
        names=counts.index,
        values=counts.values,
        hole=0.62,
        title="Répartition des types de critères",
        color_discrete_sequence=["#7C83FD", "#F38BA0"]
    )
    fig_donut.update_layout(template="plotly_white", title_x=0.02, height=350, showlegend=True)
    return fig_scatter, fig_donut


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <div class="side-brand-mark">◈</div>
            <div>
                <div class="side-brand-title">MCDM Studio</div>
                <div class="side-brand-sub">Decision App UI</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Une interface élégante pour votre projet d'aide à la décision.")

    page = st.radio("Navigation", PAGES, label_visibility="visible")

    st.markdown('<div class="sidebar-block">', unsafe_allow_html=True)
    st.markdown("### Résumé instantané")
    st.markdown(
        f"""
        <div class="sidebar-stat-grid">
            <div class="sidebar-stat"><span>Alternatives</span><strong>{len(st.session_state.df)}</strong></div>
            <div class="sidebar-stat"><span>Critères</span><strong>{len(st.session_state.df.columns)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    active_method = st.session_state.get("weight_method", "Non calculée")
    st.markdown(
        f'<div class="sidebar-chip">Méthode active : <b>{active_method}</b></div>',
        unsafe_allow_html=True,
    )
    if st.button("Réinitialiser le scénario", use_container_width=True):
        df, meta = demo_data()
        st.session_state.df = df
        st.session_state.meta = meta
        for key in ["weights", "weight_method"]:
            st.session_state.pop(key, None)
        st.session_state.weights = None
        st.session_state.weight_method = "Non calculée"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# GLOBAL HEADER
# ---------------------------------------------------------------------
hero(page)
status_ribbon()
st.markdown("<div class='spacer-16'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# PAGE 1
# ---------------------------------------------------------------------
if page == "1. Scénario & données":
    section_header(
        "Scénario de décision",
        "Un cas pédagogique réaliste, plus visuel et plus crédible qu'un simple tableau brut."
    )
    info_cols = st.columns(3)
    with info_cols[0]:
        info_card("Contexte", "Une entreprise cherche le meilleur site pour implanter un centre logistique national.", "🏭")
    with info_cols[1]:
        info_card("Décision", "Comparer les villes selon plusieurs critères contradictoires : coûts, accessibilité, proximité, impact.", "📊")
    with info_cols[2]:
        info_card("But pédagogique", "Tester différentes méthodes du cours et comparer les classements obtenus.", "🎓")

    st.markdown("<div class='spacer-8'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Vue d'ensemble", "Éditer les données", "Importer / personnaliser"])
    directions = current_directions()

    with tab1:
        left, right = st.columns([1.35, 1])
        fig_scatter, fig_donut = make_overview_figures(st.session_state.df, directions)
        with left:
            with st.container(border=True):
                st.markdown("### Aperçu visuel du scénario")
                st.caption("La taille du point reflète le niveau d'infrastructure. Le graphique sert surtout à rendre la lecture plus intuitive.")
                st.plotly_chart(fig_scatter, use_container_width=True)
        with right:
            with st.container(border=True):
                st.markdown("### Types de critères")
                st.plotly_chart(fig_donut, use_container_width=True)
            with st.container(border=True):
                st.markdown("### Critères utilisés")
                nice = st.session_state.meta.copy()
                nice["Type"] = nice["Sens"].map({"Bénéfice (+)": "Bénéfice", "Coût (-)": "Coût"})
                st.dataframe(nice[["Critère", "Type"]], use_container_width=True, hide_index=True)

    with tab2:
        left, right = st.columns([1.45, 1])
        with left:
            with st.container(border=True):
                st.markdown("### Matrice de décision")
                st.caption("Ajoutez des alternatives, modifiez les valeurs, ou supprimez des lignes directement dans le tableau.")
                edited = st.data_editor(
                    st.session_state.df,
                    use_container_width=True,
                    num_rows="dynamic",
                    key="decision_editor"
                )
                st.session_state.df = edited
        with right:
            with st.container(border=True):
                st.markdown("### Sens des critères")
                meta = st.session_state.meta.copy()
                meta = meta[meta["Critère"].isin(st.session_state.df.columns)]
                edited_meta = st.data_editor(
                    meta,
                    use_container_width=True,
                    hide_index=True,
                    disabled=["Critère"],
                    column_config={
                        "Sens": st.column_config.SelectboxColumn(
                            "Sens",
                            options=DIRECTIONS,
                            required=True
                        )
                    },
                    key="meta_editor"
                )
                st.session_state.meta = edited_meta

            with st.container(border=True):
                st.markdown("### Gérer les critères")
                new_name = st.text_input("Nom du nouveau critère", placeholder="Ex. Risque opérationnel (/10)")
                c1, c2 = st.columns(2)
                with c1:
                    new_dir = st.selectbox("Sens", DIRECTIONS, key="new_direction")
                with c2:
                    default_value = st.number_input("Valeur initiale", value=5.0, step=0.5)
                if st.button("Ajouter le critère", use_container_width=True):
                    if not new_name.strip():
                        st.warning("Donnez un nom au critère.")
                    elif new_name in st.session_state.df.columns:
                        st.warning("Ce critère existe déjà.")
                    else:
                        st.session_state.df[new_name] = float(default_value)
                        st.session_state.meta = pd.concat([
                            st.session_state.meta,
                            pd.DataFrame([{"Critère": new_name, "Sens": new_dir}])
                        ], ignore_index=True)
                        st.rerun()

                if len(st.session_state.df.columns) > 1:
                    delete_col = st.selectbox("Supprimer un critère", ["—"] + list(st.session_state.df.columns))
                    if st.button("Supprimer le critère", use_container_width=True):
                        if delete_col != "—":
                            st.session_state.df = st.session_state.df.drop(columns=[delete_col])
                            st.session_state.meta = st.session_state.meta[
                                st.session_state.meta["Critère"] != delete_col
                            ].reset_index(drop=True)
                            st.rerun()

    with tab3:
        c1, c2 = st.columns([1.25, 1])
        with c1:
            with st.container(border=True):
                st.markdown("### Importer un fichier CSV")
                st.caption("Format attendu : première colonne = noms des alternatives, autres colonnes = critères numériques.")
                up = st.file_uploader(
                    "Choisissez un CSV",
                    type=["csv"],
                    label_visibility="collapsed"
                )
                if up is not None:
                    try:
                        incoming = pd.read_csv(up)
                        incoming = incoming.set_index(incoming.columns[0])
                        incoming = incoming.apply(pd.to_numeric)
                        st.session_state.df = incoming
                        st.session_state.meta = pd.DataFrame({
                            "Critère": incoming.columns,
                            "Sens": ["Bénéfice (+)"] * len(incoming.columns)
                        })
                        st.success("CSV importé. Vérifiez maintenant le sens (+/-) de chaque critère.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Import impossible : {e}")
        with c2:
            with st.container(border=True):
                st.markdown("### Bonnes pratiques de présentation")
                st.markdown(
                    """
                    - gardez des noms d'alternatives lisibles ;
                    - mélangez critères **bénéfice** et **coût** ;
                    - évitez les colonnes vides ;
                    - préparez 8 à 12 alternatives pour une démonstration riche ;
                    - montrez ensuite la robustesse avec l'analyse de sensibilité.
                    """
                )

# ---------------------------------------------------------------------
# PAGE 2
# ---------------------------------------------------------------------
elif page == "2. Pondération":
    section_header(
        "Pondération des critères",
        "Déterminez l'importance relative des critères avant de classer les alternatives."
    )
    df = st.session_state.df.copy()
    directions = current_directions()

    if len(df.columns) < 2:
        st.error("Il faut au moins deux critères.")
        st.stop()

    c0, c1, c2, c3 = st.columns(4)
    with c0:
        info_card("AHP", "Comparaisons par paires + test de cohérence.", "⚖️")
    with c1:
        info_card("BWM", "Plus léger qu'AHP grâce à Best/Worst.", "🎯")
    with c2:
        info_card("Entropie", "Poids calculés automatiquement à partir des données.", "🧠")
    with c3:
        info_card("CRITIC", "Dispersion + corrélation entre critères.", "🔎")

    method = st.selectbox(
        "Méthode de pondération",
        [
            "AHP — Analytic Hierarchy Process",
            "BWM — Best-Worst Method",
            "Entropie",
            "CRITIC",
            "Poids égaux (outil pratique)"
        ]
    )

    weights = None
    with st.container(border=True):
        if method.startswith("AHP"):
            st.markdown("### Saisie AHP")
            st.caption(
                "Échelle de Saaty 1 à 9. Une valeur supérieure à 1 favorise le premier critère ; une valeur inférieure à 1 favorise le second."
            )
            n = len(df.columns)
            if n > 10:
                st.warning(
                    "Le tableau RI du cours va jusqu'à 10 critères. Réduisez le nombre de critères pour utiliser AHP strictement selon le support."
                )
                st.stop()

            criteria = list(df.columns)
            A = np.ones((n, n), dtype=float)
            cols = st.columns(3)
            counter = 0
            for i in range(n):
                for j in range(i + 1, n):
                    c = cols[counter % 3]
                    with c:
                        choice = st.selectbox(
                            f"{criteria[i]} vs {criteria[j]}",
                            options=[1/9,1/8,1/7,1/6,1/5,1/4,1/3,1/2,1,2,3,4,5,6,7,8,9],
                            index=8,
                            key=f"ahp_{i}_{j}",
                            format_func=lambda x: f"{x:g}"
                        )
                    A[i, j] = float(choice)
                    A[j, i] = 1 / float(choice)
                    counter += 1

            w, info = ahp_weights(A)
            weights = pd.Series(w, index=criteria, name="Poids")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("λmax", f"{info['lambda_max']:.4f}")
            m2.metric("CI", f"{info['CI']:.4f}")
            m3.metric("RI", f"{info['RI']:.2f}")
            m4.metric("CR", f"{info['CR']:.4f}")
            if info["CR"] < 0.1:
                st.success("CR < 0,1 : les jugements sont cohérents.")
            else:
                st.warning("CR ≥ 0,1 : pensez à revoir certaines comparaisons.")
            with st.expander("Voir la matrice AHP"):
                st.dataframe(pd.DataFrame(A, index=criteria, columns=criteria), use_container_width=True)

        elif method.startswith("BWM"):
            st.markdown("### Saisie BWM")
            criteria = list(df.columns)
            a, b = st.columns(2)
            with a:
                best = st.selectbox("Critère Best", criteria)
            with b:
                worst = st.selectbox("Critère Worst", [c for c in criteria if c != best])

            st.markdown("#### Best-to-Others (BO)")
            bo = {}
            cols = st.columns(4)
            for i, crit in enumerate(criteria):
                if crit == best:
                    bo[crit] = 1
                    cols[i % 4].number_input(f"{best} → {crit}", min_value=1, max_value=1, value=1, disabled=True)
                else:
                    bo[crit] = cols[i % 4].slider(f"{best} → {crit}", 1, 9, 3, key=f"bo_{crit}")

            st.markdown("#### Others-to-Worst (OW)")
            ow = {}
            cols = st.columns(4)
            for i, crit in enumerate(criteria):
                if crit == worst:
                    ow[crit] = 1
                    cols[i % 4].number_input(f"{crit} → {worst}", min_value=1, max_value=1, value=1, disabled=True)
                else:
                    ow[crit] = cols[i % 4].slider(f"{crit} → {worst}", 1, 9, 3, key=f"ow_{crit}")
            try:
                weights, xi = bwm_weights(criteria, best, worst, bo, ow)
                st.metric("ξ*", f"{xi:.4f}")
                st.caption("Plus ξ* est proche de 0, plus les jugements sont cohérents.")
            except Exception as e:
                st.error(str(e))

        elif method == "Entropie":
            st.markdown("### Calcul Entropie")
            weights, R, details = entropy_weights(df, directions)
            st.dataframe(details.round(5), use_container_width=True)
            with st.expander("Voir la matrice normalisée"):
                st.dataframe(R.round(5), use_container_width=True)

        elif method == "CRITIC":
            st.markdown("### Calcul CRITIC")
            weights, R, corr, details = critic_weights(df, directions)
            st.dataframe(details.round(5), use_container_width=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Corrélations entre critères",
                color_continuous_scale=[[0.0, "#FFD8E8"], [0.5, "#FFF7FB"], [1.0, "#B7D7FF"]]
            )
            fig.update_layout(template="plotly_white", title_x=0.02, height=480)
            st.plotly_chart(fig, use_container_width=True)
        else:
            weights = equal_weights(df.columns)
            st.info("Chaque critère reçoit le même poids. Utile pour une démonstration rapide ou comme référence.")

    if weights is not None:
        st.markdown("<div class='spacer-8'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.5])
        with c1:
            with st.container(border=True):
                st.markdown("### Tableau des poids")
                st.dataframe(weights.to_frame().round(5), use_container_width=True)
        with c2:
            with st.container(border=True):
                st.plotly_chart(weight_chart(weights), use_container_width=True)

        st.session_state.weights = weights
        st.session_state.weight_method = method
        st.success("Les poids sont enregistrés et seront réutilisés dans les étapes suivantes.")

# ---------------------------------------------------------------------
# PAGE 3
# ---------------------------------------------------------------------
elif page == "3. Classement":
    section_header(
        "Classement des alternatives",
        "Appliquez la méthode choisie et ouvrez l’onglet Étapes de calcul pour voir la normalisation et les calculs intermédiaires."
    )

    if st.session_state.get("weights") is None:
        st.warning("Commencez par calculer les poids dans l'étape 2 — Pondération.")
        st.stop()

    df = st.session_state.df.copy()
    directions = current_directions()
    weights = st.session_state.weights.reindex(df.columns).fillna(0)
    weights = weights / weights.sum()

    left, right = st.columns([1.2, 0.8])
    with left:
        method = st.selectbox(
            "Méthode de classement",
            ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]
        )
    with right:
        page_badge(
            f"Pondération active : {st.session_state.get('weight_method', '—')}",
            "lavender"
        )

    lam, v = 0.5, 0.5
    controls = st.columns(2)
    with controls[0]:
        if method == "WASPAS":
            lam = st.slider(
                "λ — poids de WSM dans WASPAS",
                0.0, 1.0, 0.5, 0.05
            )
    with controls[1]:
        if method == "VIKOR":
            v = st.slider(
                "v — poids de l'utilité de groupe",
                0.0, 1.0, 0.5, 0.05
            )

    try:
        # -------------------------------------------------------------
        # CALCUL PRINCIPAL
        # -------------------------------------------------------------
        if method == "WSM":
            score, R = wsm(df, directions, weights)
            result = rank_table(score, True)
            higher = True

        elif method == "WPM":
            score, R = wpm(df, directions, weights)
            result = rank_table(score, True)
            higher = True

        elif method == "WASPAS":
            score, q1, q2, R = waspas(df, directions, weights, lam)
            result = rank_table(score, True)
            result["WSM"] = q1
            result["WPM"] = q2
            higher = True

        elif method == "TOPSIS":
            score, details, V = topsis(df, directions, weights)
            result = rank_table(score, True).join(details[["S+", "S-"]])
            higher = True

            # Matrice normalisée TOPSIS, affichée ensuite dans les étapes.
            X_topsis = df.to_numpy(dtype=float)
            den_topsis = np.sqrt((X_topsis ** 2).sum(axis=0))
            den_topsis = np.where(den_topsis <= 1e-12, 1.0, den_topsis)
            R_topsis = pd.DataFrame(
                X_topsis / den_topsis,
                index=df.index,
                columns=df.columns
            )

            # Solutions idéales positive / négative à partir de V.
            ideal_plus = {}
            ideal_minus = {}
            for col in df.columns:
                if directions[col] == "Bénéfice (+)":
                    ideal_plus[col] = V[col].max()
                    ideal_minus[col] = V[col].min()
                else:
                    ideal_plus[col] = V[col].min()
                    ideal_minus[col] = V[col].max()

            ideals_topsis = pd.DataFrame(
                [ideal_plus, ideal_minus],
                index=["Idéal positif A+", "Idéal négatif A-"]
            )

        else:  # VIKOR
            score, details = vikor(df, directions, weights, v)
            result = rank_table(score, False).join(details[["S", "R"]])
            higher = False

            # Meilleures et pires valeurs utilisées par VIKOR.
            best_values = {}
            worst_values = {}
            for col in df.columns:
                if directions[col] == "Bénéfice (+)":
                    best_values[col] = df[col].max()
                    worst_values[col] = df[col].min()
                else:
                    best_values[col] = df[col].min()
                    worst_values[col] = df[col].max()

            ideal_vikor = pd.DataFrame(
                [best_values, worst_values],
                index=["Meilleure valeur x*", "Pire valeur x-"]
            )

            # Rapport normalisé utilisé dans S et R :
            # (x*_j - x_ij) / (x*_j - x-_j)
            gap_vikor = pd.DataFrame(
                index=df.index,
                columns=df.columns,
                dtype=float
            )
            for col in df.columns:
                denom = best_values[col] - worst_values[col]
                if abs(denom) <= 1e-12:
                    gap_vikor[col] = 0.0
                else:
                    gap_vikor[col] = (
                        best_values[col] - df[col].astype(float)
                    ) / denom

            weighted_gap_vikor = gap_vikor.mul(weights, axis=1)

        # -------------------------------------------------------------
        # RÉSULTAT VISUEL
        # -------------------------------------------------------------
        podium_cards(result)
        best = result.index[0]

        st.markdown(
            f'<div class="highlight-band">'
            f'<strong>Meilleure alternative selon {method}</strong>'
            f'<span>{best}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        tab1, tab2, tab3 = st.tabs([
            "Résultats",
            "Étapes de calcul",
            "Téléchargement"
        ])

        # -------------------------------------------------------------
        # ONGLET 1 — RÉSULTATS
        # -------------------------------------------------------------
        with tab1:
            c1, c2 = st.columns([1.05, 1.35])

            with c1:
                with st.container(border=True):
                    st.markdown("### Tableau du classement")
                    st.dataframe(
                        result.round(5),
                        use_container_width=True
                    )

            with c2:
                with st.container(border=True):
                    st.plotly_chart(
                        pastel_bar(
                            score,
                            f"Scores — {method}",
                            higher_better=higher
                        ),
                        use_container_width=True
                    )

        # -------------------------------------------------------------
        # ONGLET 2 — ÉTAPES DE CALCUL
        # -------------------------------------------------------------
        with tab2:
            st.markdown("## Étapes détaillées du calcul")
            st.caption(
                "La normalisation n'est pas identique pour toutes les méthodes. "
                "L'application affiche donc la transformation réellement utilisée par la méthode sélectionnée."
            )

            # Étape commune : matrice initiale
            with st.container(border=True):
                st.markdown("### Étape 1 — Matrice de décision initiale X")
                st.dataframe(
                    df.round(4),
                    use_container_width=True
                )

            with st.container(border=True):
                st.markdown("### Poids des critères utilisés")
                st.dataframe(
                    weights.to_frame("Poids").round(4),
                    use_container_width=True
                )

            # ---------------------------------------------------------
            # WSM
            # ---------------------------------------------------------
            if method == "WSM":
                with st.container(border=True):
                    st.markdown("### Étape 2 — Normalisation WSM")
                    st.markdown(
                        "Pour un critère **bénéfice (+)** : "
                        r"$r_{ij}=x_{ij}/\max(x_j)$.  "
                        "Pour un critère **coût (-)** : "
                        r"$r_{ij}=\min(x_j)/x_{ij}$."
                    )
                    st.dataframe(R.round(4), use_container_width=True)

                weighted_wsm = R.mul(weights, axis=1)
                with st.container(border=True):
                    st.markdown("### Étape 3 — Matrice normalisée pondérée")
                    st.latex(r"w_j r_{ij}")
                    st.dataframe(weighted_wsm.round(4), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### Étape 4 — Score WSM et classement")
                    st.latex(r"Q_i = \sum_{j=1}^{n} w_j r_{ij}")
                    st.dataframe(result.round(4), use_container_width=True)

            # ---------------------------------------------------------
            # WPM
            # ---------------------------------------------------------
            elif method == "WPM":
                with st.container(border=True):
                    st.markdown("### Étape 2 — Normalisation WPM")
                    st.markdown(
                        "La normalisation est la même que pour WSM : "
                        "bénéfice → x/max(x), coût → min(x)/x."
                    )
                    st.dataframe(R.round(4), use_container_width=True)

                powered_wpm = pd.DataFrame(
                    np.power(
                        R.clip(lower=1e-12).to_numpy(dtype=float),
                        weights.to_numpy(dtype=float)
                    ),
                    index=R.index,
                    columns=R.columns
                )

                with st.container(border=True):
                    st.markdown("### Étape 3 — Termes multiplicatifs pondérés")
                    st.latex(r"(r_{ij})^{w_j}")
                    st.dataframe(powered_wpm.round(4), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### Étape 4 — Score WPM et classement")
                    st.latex(r"Q_i = \prod_{j=1}^{n}(r_{ij})^{w_j}")
                    st.dataframe(result.round(4), use_container_width=True)

            # ---------------------------------------------------------
            # WASPAS
            # ---------------------------------------------------------
            elif method == "WASPAS":
                with st.container(border=True):
                    st.markdown("### Étape 2 — Normalisation WASPAS")
                    st.markdown(
                        "WASPAS utilise ici la même matrice normalisée que WSM et WPM."
                    )
                    st.dataframe(R.round(4), use_container_width=True)

                c_wsm, c_wpm = st.columns(2)
                with c_wsm:
                    with st.container(border=True):
                        st.markdown("### Étape 3 — Partie WSM")
                        st.latex(r"Q_i^{(1)}=\sum_j w_jr_{ij}")
                        st.dataframe(
                            q1.to_frame("Q WSM").round(4),
                            use_container_width=True
                        )

                with c_wpm:
                    with st.container(border=True):
                        st.markdown("### Étape 4 — Partie WPM")
                        st.latex(r"Q_i^{(2)}=\prod_j(r_{ij})^{w_j}")
                        st.dataframe(
                            q2.to_frame("Q WPM").round(4),
                            use_container_width=True
                        )

                with st.container(border=True):
                    st.markdown("### Étape 5 — Agrégation WASPAS")
                    st.latex(
                        r"Q_i=\lambda Q_i^{(1)}+(1-\lambda)Q_i^{(2)}"
                    )
                    st.write(f"λ = {lam:.2f}")
                    st.dataframe(result.round(4), use_container_width=True)

            # ---------------------------------------------------------
            # TOPSIS
            # ---------------------------------------------------------
            elif method == "TOPSIS":
                with st.container(border=True):
                    st.markdown("### Étape 2 — Normalisation vectorielle TOPSIS")
                    st.latex(
                        r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^{2}}}"
                    )
                    st.dataframe(R_topsis.round(4), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### Étape 3 — Matrice normalisée pondérée V")
                    st.latex(r"v_{ij}=w_jr_{ij}")
                    st.dataframe(V.round(4), use_container_width=True)

                with st.container(border=True):
                    st.markdown("### Étape 4 — Solutions idéales")
                    st.markdown(
                        "Pour un critère bénéfice, l'idéal positif est le maximum. "
                        "Pour un critère coût, l'idéal positif est le minimum."
                    )
                    st.dataframe(
                        ideals_topsis.round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 5 — Distances aux idéaux")
                    st.latex(
                        r"S_i^+=\sqrt{\sum_j(v_{ij}-v_j^*)^2}"
                    )
                    st.latex(
                        r"S_i^-=\sqrt{\sum_j(v_{ij}-v_j^-)^2}"
                    )
                    st.dataframe(
                        details[["S+", "S-"]].round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 6 — Coefficient de proximité et classement")
                    st.latex(
                        r"RC_i=\frac{S_i^-}{S_i^-+S_i^+}"
                    )
                    st.dataframe(result.round(4), use_container_width=True)

            # ---------------------------------------------------------
            # VIKOR
            # ---------------------------------------------------------
            elif method == "VIKOR":
                with st.container(border=True):
                    st.markdown("### Étape 2 — Meilleures et pires valeurs")
                    st.dataframe(
                        ideal_vikor.round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 3 — Rapport normalisé VIKOR")
                    st.latex(
                        r"d_{ij}=\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}"
                    )
                    st.dataframe(
                        gap_vikor.round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 4 — Termes pondérés")
                    st.latex(r"w_jd_{ij}")
                    st.dataframe(
                        weighted_gap_vikor.round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 5 — Calcul de S et R")
                    st.latex(
                        r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}"
                    )
                    st.latex(
                        r"R_i=\max_j\left[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\right]"
                    )
                    st.dataframe(
                        details[["S", "R"]].round(4),
                        use_container_width=True
                    )

                with st.container(border=True):
                    st.markdown("### Étape 6 — Calcul de Q et classement")
                    st.latex(
                        r"Q_i=v\frac{S_i-S^*}{S^- - S^*}"
                        r"+(1-v)\frac{R_i-R^*}{R^- - R^*}"
                    )
                    st.write(f"v = {v:.2f}")
                    st.dataframe(
                        details.round(4),
                        use_container_width=True
                    )
                    st.caption(
                        "Pour VIKOR, plus Q est petit, meilleure est l'alternative."
                    )

        # -------------------------------------------------------------
        # ONGLET 3 — TÉLÉCHARGEMENT
        # -------------------------------------------------------------
        with tab3:
            st.markdown("### Exporter les résultats")
            st.download_button(
                "Télécharger le classement CSV",
                result.to_csv().encode("utf-8-sig"),
                file_name=f"classement_{method.lower()}.csv",
                mime="text/csv"
            )
            st.caption(
                "Le fichier contient les scores et le classement final des alternatives."
            )

    except Exception as e:
        st.error(str(e))

# ---------------------------------------------------------------------
# PAGE 4
# ---------------------------------------------------------------------
elif page == "4. Comparaison":
    section_header(
        "Comparer toutes les méthodes",
        "Une vraie application d'aide à la décision doit aussi montrer les écarts et convergences entre méthodes."
    )

    if st.session_state.get("weights") is None:
        st.warning("Calculez d'abord les poids dans l'étape 2.")
        st.stop()

    df = st.session_state.df.copy()
    directions = current_directions()
    weights = st.session_state.weights.reindex(df.columns).fillna(0)
    weights = weights / weights.sum()

    c1, c2 = st.columns(2)
    with c1:
        lam = st.slider("λ pour WASPAS", 0.0, 1.0, 0.5, 0.05, key="cmp_lam")
    with c2:
        v = st.slider("v pour VIKOR", 0.0, 1.0, 0.5, 0.05, key="cmp_v")

    try:
        ranks = compare_rankings(df, directions, weights, lam, v)
        ranks["Rang moyen"] = ranks.mean(axis=1)
        ranks = ranks.sort_values("Rang moyen")

        winners = {m: ranks[m].idxmin() for m in ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]}
        counts = pd.Series(winners).value_counts()
        consensus = counts.index[0]

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Consensus principal", consensus, "alternative la plus souvent première")
        with c2:
            kpi_card("Méthodes alignées", f"{counts.iloc[0]} / 5", "nombre de premières places")
        with c3:
            kpi_card("Top moyenne", ranks.index[0], "meilleur rang moyen")

        tab1, tab2, tab3 = st.tabs(["Classements", "Consensus", "Spearman"])
        with tab1:
            with st.container(border=True):
                st.markdown("### Tableau des rangs")
                st.dataframe(ranks.round(3), use_container_width=True)
        with tab2:
            l, r = st.columns([1, 1.4])
            with l:
                with st.container(border=True):
                    st.markdown("### Première place par méthode")
                    st.dataframe(pd.DataFrame({"Méthode": winners.keys(), "1er rang": winners.values()}), use_container_width=True, hide_index=True)
            with r:
                with st.container(border=True):
                    fig = px.bar(
                        x=counts.index,
                        y=counts.values,
                        labels={"x": "Alternative", "y": "Nombre de premières places"},
                        title="Nombre de méthodes qui classent l'alternative première"
                    )
                    fig.update_traces(marker_color=[PASTELS[i % len(PASTELS)] for i in range(len(counts))])
                    fig.update_layout(template="plotly_white", title_x=0.02, height=390)
                    st.plotly_chart(fig, use_container_width=True)
        with tab3:
            corr = spearman_matrix(ranks[["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"]])
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
                color_continuous_scale=[[0.0, "#FFD3DE"], [0.5, "#FFFDFD"], [1.0, "#BFD9FF"]],
                title="Matrice de Spearman"
            )
            fig.update_layout(template="plotly_white", title_x=0.02, height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Télécharger la comparaison CSV",
            ranks.to_csv().encode("utf-8-sig"),
            file_name="comparaison_methodes.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(str(e))

# ---------------------------------------------------------------------
# PAGE 5
# ---------------------------------------------------------------------
elif page == "5. Sensibilité":
    section_header(
        "Analyse de sensibilité",
        "Montrez la robustesse de votre résultat : si les poids changent, le classement reste-t-il stable ?"
    )

    if st.session_state.get("weights") is None:
        st.warning("Calculez d'abord les poids dans l'étape 2.")
        st.stop()

    df = st.session_state.df.copy()
    directions = current_directions()
    base = st.session_state.weights.reindex(df.columns).fillna(0)
    base = base / base.sum()

    c1, c2 = st.columns(2)
    with c1:
        criterion = st.selectbox("Critère à faire varier", list(df.columns))
    with c2:
        method = st.selectbox("Méthode observée", ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR"])

    lam = 0.5
    v = 0.5
    if method == "WASPAS":
        lam = st.slider("λ WASPAS", 0.0, 1.0, 0.5, 0.05, key="sens_lam")
    if method == "VIKOR":
        v = st.slider("v VIKOR", 0.0, 1.0, 0.5, 0.05, key="sens_v")

    grid = np.linspace(0.05, 0.70, 14)
    records = []
    for target_w in grid:
        w = base.copy()
        other_cols = [c for c in w.index if c != criterion]
        other_sum = base[other_cols].sum()
        w[criterion] = target_w
        if other_sum <= 1e-12:
            w[other_cols] = (1 - target_w) / len(other_cols)
        else:
            w[other_cols] = base[other_cols] / other_sum * (1 - target_w)
        if method == "WSM":
            scores, _ = wsm(df, directions, w)
            higher = True
        elif method == "WPM":
            scores, _ = wpm(df, directions, w)
            higher = True
        elif method == "WASPAS":
            scores, *_ = waspas(df, directions, w, lam)
            higher = True
        elif method == "TOPSIS":
            scores, *_ = topsis(df, directions, w)
            higher = True
        else:
            scores, _ = vikor(df, directions, w, v)
            higher = False
        ranks = scores.rank(ascending=not higher, method="min")
        for alt, rank in ranks.items():
            records.append({"Poids": target_w, "Alternative": alt, "Rang": rank})

    sens = pd.DataFrame(records)
    fig = px.line(
        sens,
        x="Poids", y="Rang", color="Alternative", markers=True,
        title=f"Stabilité des rangs — {method}", color_discrete_sequence=PASTELS
    )
    fig.update_yaxes(autorange="reversed", dtick=1)
    fig.update_layout(template="plotly_white", title_x=0.02, height=620)

    l, r = st.columns([1.4, 1])
    with l:
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)
    with r:
        winners = (
            sens.loc[sens.groupby("Poids")["Rang"].idxmin(), ["Poids", "Alternative", "Rang"]]
            .sort_values("Poids")
        )
        with st.container(border=True):
            st.markdown("### Alternative classée première")
            st.dataframe(winners, use_container_width=True, hide_index=True)
            stable_count = winners["Alternative"].value_counts().max()
            dominant = winners["Alternative"].value_counts().idxmax()
            st.markdown(
                f"<div class='sidebar-chip'>Alternative dominante : <b>{dominant}</b> — première sur {stable_count} niveaux de poids.</div>",
                unsafe_allow_html=True
            )

# ---------------------------------------------------------------------
# PAGE 6
# ---------------------------------------------------------------------
else:
    section_header(
        "Guide des méthodes du cours",
        "Une page plus propre pour expliquer votre application au professeur ou dans la présentation."
    )

    methods = [
        ("AHP", "Pondération subjective", "Comparaisons par paires, échelle 1–9, puis test CR < 0,1."),
        ("BWM", "Pondération subjective", "Choisir Best/Worst puis résoudre un programme linéaire minimisant ξ."),
        ("Entropie", "Pondération objective", "Plus un critère discrimine les alternatives, plus son poids est important."),
        ("CRITIC", "Pondération objective", "Combine dispersion σ et faible corrélation avec les autres critères."),
        ("WSM", "Classement", "Somme pondérée des performances normalisées."),
        ("WPM", "Classement", "Produit pondéré des performances normalisées."),
        ("WASPAS", "Classement", "Combine WSM et WPM grâce au paramètre λ."),
        ("TOPSIS", "Classement", "Cherche l'alternative proche de l'idéal positif et loin de l'idéal négatif."),
        ("VIKOR", "Classement", "Recherche une solution de compromis entre utilité de groupe S et regret R."),
        ("Spearman", "Comparaison", "Mesure la similarité entre deux classements."),
        ("Sensibilité", "Robustesse", "Fait varier les poids pour vérifier si le classement reste stable."),
    ]
    guide = pd.DataFrame(methods, columns=["Méthode", "Rôle", "Idée essentielle"])
    a, b = st.columns([1.1, 0.9])
    with a:
        with st.container(border=True):
            st.dataframe(guide, use_container_width=True, hide_index=True)
    with b:
        with st.container(border=True):
            st.markdown("### Portée de cette version")
            st.markdown(
                """
                - Implémentées : **AHP, BWM, Entropie, CRITIC, WSM, WPM, WASPAS, TOPSIS, VIKOR**.
                - Ajoutées : **corrélation de Spearman** et **analyse de sensibilité**.
                - Non implémentées dans cette version : **DEMATEL, SWARA, ELECTRE, PROMETHEE, ANP, Fuzzy MCDM**.
                """
            )
            st.markdown("### Logique générale")
            st.markdown("**Données → sens des critères → pondération → classement → comparaison → robustesse.**")

st.markdown("<div class='app-footer'>Conçu pour un projet Streamlit MCDM • design pastel & démonstration pédagogique.</div>", unsafe_allow_html=True)
