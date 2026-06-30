import io

import pandas as pd
import streamlit as st

from assign_classes_module import (
    AVEC_FIELDS,
    SANS_FIELDS,
    DEFAULT_WEIGHTS,
    load_data,
    compute_capacities,
    build_allowed,
    describe_no_class,
    solve,
)
from report_module import (
    reorder_columns,
    build_summary,
    compute_constraints,
    build_workbook,
)
from update_classes_module import validate_and_update_workbook

st.title("🎓 Constitution Automatique des Classes")

st.markdown("""
## 📄 Instructions - Format du fichier Excel

Le fichier Excel doit comporter **2 feuilles** :

### 🧑‍🎓 Feuille `liste` : Élèves à affecter

Colonnes obligatoires :

- `Elèves à affecter` : nom unique de l'élève
- `Genre` : `F` (fille) ou `G` (garçon)
- `por`, `lat`, `pp`, `tech`, `cat i` : `1` si l'élève suit cette structure, sinon `0`
- `pap` : `1` si l'élève bénéficie d'un PAP, sinon `0` (équilibré + compté par classe)
- `Niveau` : niveau scolaire (ex : 1, 2, 3)
- `Comportement` : de 1 bon à 3 difficile
- `avec1`, `avec2` *(facultatif)* : noms d'élèves avec qui il souhaite être
- `sans1` … `sans5` *(facultatif)* : noms d'élèves à éviter (prioritaires sur les `avec`)

### 🏫 Feuille `classes` : Classes disponibles

Colonnes obligatoires :

- `Nom` : nom de la classe (ex: A, B, C)
- `por`, `lat`, `pp`, `tech`, `cat i` : `1` si la classe propose cette structure (sinon vide)
- Un élève marqué `por`/`lat`/`tech`/`cat i` **doit** être placé dans une classe
  proposant cette structure (contrainte dure ; la classe peut être complétée par
  d'autres élèves).
- `pp` décrit une classe prépa métiers, elle n'est donc constituée que d'élèves pp.
- `capacité` : plafond **dur** d'élèves — la classe ne le dépassera jamais
  (facultatif ; peut être renseigné sur toutes les classes ou aucune)

---

💡 Vous pouvez télécharger comme modèle le fichier ci-dessous.
""")

with open("Liste.xlsx", "rb") as f:
    st.download_button(
        label="📥 Télécharger le modèle Liste.xlsx",
        data=f,
        file_name="Liste.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

st.markdown("""
## 📤 Structure du fichier Excel de sortie

Une fois le traitement terminé, l'application génère un fichier
`assignments_desanonymiser.xlsm` (macro-actif) comportant plusieurs feuilles :
`Classes` (affectations), `Impossibilites` (vœux non respectés), `Contraintes`
(vœux pris en compte), `Tableau` (matrice de répartition) et `Dashboards`
(statistiques par classe).

La macro **« Desanonymiser »** y est déjà incrustée : ouvrez le fichier, collez
l'onglet `diccionari` (colonnes `nom_real` / `id_anonim`) puis lancez la macro
pour rétablir les vrais noms.
""")


@st.cache_data(show_spinner="Calcul de l'affectation en cours…")
def run_assignment(file_bytes: bytes, weights: dict):
    """Calcule l'affectation. Mise en cache sur le contenu du fichier ET les
    poids : un nouveau fichier ou un poids modifié relance automatiquement le
    calcul."""
    df_students = pd.read_excel(io.BytesIO(file_bytes), sheet_name="liste")
    df_students = df_students.rename(columns={"Elèves à affecter": "student"})
    df_classes = pd.read_excel(io.BytesIO(file_bytes), sheet_name="classes")

    students_df, classes_df, override_map = load_data(df_students.copy(), df_classes)
    classes_df = compute_capacities(students_df, classes_df, override_map)
    allowed = build_allowed(students_df, classes_df)

    total_students = len(students_df)
    total_capacity = int(classes_df["capacity"].sum())
    no_class = [s for s, lst in allowed.items() if not lst]

    # Conflits d'options : un élève demande des structures qu'aucune classe ne
    # combine. On arrête avec un message clair invitant à corriger le fichier.
    if no_class:
        problems = describe_no_class(students_df, classes_df)
        lines = "\n".join(
            f"• {s} (options : {', '.join(problems[s]['options']) or 'aucune'}) "
            f"→ {problems[s]['reason']}"
            for s in no_class
        )
        raise RuntimeError(
            "Certains élèves n'ont aucune classe possible (structures incompatibles).\n"
            "Corrigez le fichier d'entrée ou ajustez les structures de classe :\n"
            f"{lines}"
        )

    if total_capacity < total_students:
        raise RuntimeError(
            f"Capacité totale insuffisante : {total_capacity} < {total_students}"
        )

    assignment, broken = solve(allowed, classes_df, students_df, weights=weights)

    students = students_df.copy()
    students["classe"] = [assignment[s] for s in students.index]
    students.reset_index(inplace=True)
    students_sorted = reorder_columns(students.sort_values(["classe", "student"]))

    return {
        "students_sorted": students_sorted,
        "students_raw": df_students,
        "broken": broken,
        "total_students": total_students,
        "total_capacity": total_capacity,
        "no_class": no_class,
    }


def weights_sidebar() -> dict:
    """Panneau de pondération des règles. Renvoie un dict {clé: poids} prêt à
    passer à `solve`. Les défauts proviennent de `DEFAULT_WEIGHTS`."""
    st.sidebar.header("⚖️ Pondération des règles")
    st.sidebar.caption(
        "Plus le poids est élevé, plus la règle est prioritaire. "
        "Les écarts d'équité sont en pour-mille (0–1000)."
    )
    weights = {}

    with st.sidebar.expander("Vœux sociaux", expanded=True):
        sans = st.number_input(
            "Séparer (sans) — par vœu cassé", 0, 5000,
            int(DEFAULT_WEIGHTS["sans1"]), 50,
            help="Coût si deux élèves « sans » se retrouvent ensemble.",
        )
        avec = st.number_input(
            "Regrouper (avec) — par vœu cassé", 0, 5000,
            int(DEFAULT_WEIGHTS["avec1"]), 50,
            help="Coût si deux élèves « avec » ne sont pas ensemble.",
        )
    for f in SANS_FIELDS:
        weights[f] = sans
    for f in AVEC_FIELDS:
        weights[f] = avec

    with st.sidebar.expander("Origine"):
        weights["origin_singleton"] = st.number_input(
            "Éviter un élève seul de son origine", 0, 5000,
            int(DEFAULT_WEIGHTS["origin_singleton"]), 50,
        )

    equity_labels = {
        "fill": "Remplissage des classes",
        "gender": "Genre (filles/garçons)",
        "level1": "Niveau 1", "level2": "Niveau 2", "level3": "Niveau 3",
        "comp1": "Comportement 1", "comp2": "Comportement 2", "comp3": "Comportement 3",
        "por": "POR", "lat": "LAT", "pap": "PAP",
    }
    with st.sidebar.expander("Équité (affinage)"):
        st.caption("Multiplicateur. Coût = poids × écart (‰, 0–1000).")
        for key, label in equity_labels.items():
            weights[key] = st.number_input(
                label, 0, 50, int(DEFAULT_WEIGHTS[key]), 1, key=f"w_{key}",
                help="Multiplicateur. Coût = poids × écart de remplissage (‰).",
            )

    return weights


# --- Interface Streamlit ---
st.sidebar.header("Chargement du fichier")
input_file = st.sidebar.file_uploader(
    "Votre fichier .xlsx (onglets 'liste' et 'classes')", type="xlsx"
)

weights = weights_sidebar()

if not input_file:
    st.info("⬅️ Importez le fichier .xlsx pour lancer l'affectation.")
    st.stop()

file_bytes = input_file.getvalue()

# Signature des entrées : sert à détecter qu'un calcul affiché est périmé
# (fichier ou pondérations modifiés depuis le dernier « Lancer l'affectation »).
inputs_sig = (hash(file_bytes), tuple(sorted(weights.items())))

run_clicked = st.sidebar.button("🚀 Lancer l'affectation", type="primary")
if run_clicked:
    try:
        st.session_state["result"] = run_assignment(file_bytes, weights)
        st.session_state["result_sig"] = inputs_sig
    except Exception as e:
        st.session_state.pop("result", None)
        st.error(f"❌ Une erreur s'est produite : {e}")
        st.stop()

result = st.session_state.get("result")
if result is None:
    st.info("⬅️ Réglez les pondérations puis cliquez sur « 🚀 Lancer l'affectation ».")
    st.stop()

if st.session_state.get("result_sig") != inputs_sig:
    st.warning(
        "⚠️ Fichier ou pondérations modifiés depuis le dernier calcul. "
        "Cliquez sur « 🚀 Lancer l'affectation » pour actualiser."
    )

st.write(
    f"Total élèves = {result['total_students']}, "
    f"capacité totale = {result['total_capacity']}"
)
for s in result["no_class"]:
    st.warning(f"⚠️ Élève sans affectation possible : {s}")

students_sorted = result["students_sorted"]
students_raw = result["students_raw"]

st.write("### Résultats de l'affectation")

# Feuilles de sortie via le module partagé.
summary = build_summary(students_sorted)
assign_map = dict(zip(students_sorted["student"].astype(str), students_sorted["classe"]))
constraints_df, impossibilites_df = compute_constraints(students_raw, assign_map)
buffer = build_workbook(students_sorted, constraints_df, impossibilites_df, summary)

st.download_button(
    "📥 Télécharger le fichier Excel",
    data=buffer,
    file_name="assignments_desanonymiser.xlsm",
    mime="application/vnd.ms-excel.sheet.macroEnabled.12",
    key="download_initial",
)

st.markdown("""
### ✏️ Modifier les affectations manuellement

Vous pouvez modifier les classes attribuées aux élèves directement dans le tableau ci-dessous.
Une fois vos modifications terminées, cliquez sur **🔁 Rafraîchir et vérifier les contraintes** pour
mettre à jour toutes les feuilles et télécharger le nouveau fichier Excel.
""")

edited_df = st.data_editor(
    students_sorted,
    column_config={
        "classe": st.column_config.SelectboxColumn(
            "Classe", options=sorted(students_sorted["classe"].unique().tolist())
        )
    },
    use_container_width=True,
)

if st.button("🔁 Rafraîchir et vérifier les contraintes"):
    results = validate_and_update_workbook(edited_df, students_raw)
    st.success("📦 Fichier 'assignments.xlsx' mis à jour avec succès.")
    st.download_button(
        "📥 Télécharger le fichier Excel mis à jour",
        data=results["buffer"],
        file_name="assignments_desanonymiser.xlsm",
        mime="application/vnd.ms-excel.sheet.macroEnabled.12",
        key="download_updated",
    )
