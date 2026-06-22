import io

import pandas as pd
import streamlit as st

from assign_classes_module import (
    load_data,
    compute_capacities,
    build_allowed,
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
- `por`, `lat`, `pp` : `1` si l'élève souhaite suivre cette option, sinon `0`
- `Niveau` : niveau scolaire (ex : 1, 2, 3)
- `Comportement` : de 1 bon à 3 difficile
- `avec1`, `avec2` *(facultatif)* : noms d'élèves avec qui il souhaite être
- `sans1`, `sans2` *(facultatif)* : noms d'élèves à éviter

### 🏫 Feuille `classes` : Classes disponibles

Colonnes obligatoires :

- `Nom` : nom de la classe (ex: A, B, C)
- `por`, `lat`, `pp` : `1` si la classe permet cette option (sinon vide)
- `pp` décrit une classe prépa métiers, elle n'est donc constituée que d'élèves pp.
- `capacité` : nombre maximal d'élèves (facultatif)

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

Une fois le traitement terminé, l'application génère un fichier Excel comportant
plusieurs feuilles : `Classes` (affectations), `Impossibilites` (vœux non
respectés), `Contraintes` (vœux pris en compte), `Tableau` (matrice de
répartition) et `Dashboards` (statistiques par classe).
""")


@st.cache_data(show_spinner="Calcul de l'affectation en cours…")
def run_assignment(file_bytes: bytes):
    """Calcule l'affectation. Mise en cache sur le contenu du fichier :
    un nouveau fichier relance automatiquement le calcul."""
    df_students = pd.read_excel(io.BytesIO(file_bytes), sheet_name="liste")
    df_students = df_students.rename(columns={"Elèves à affecter": "student"})
    df_classes = pd.read_excel(io.BytesIO(file_bytes), sheet_name="classes")

    students_df, classes_df, override_map = load_data(df_students.copy(), df_classes)
    classes_df = compute_capacities(students_df, classes_df, override_map)
    allowed = build_allowed(students_df, classes_df)

    total_students = len(students_df)
    total_capacity = int(classes_df["capacity"].sum())
    no_class = [s for s, lst in allowed.items() if not lst]

    if total_capacity < total_students:
        raise RuntimeError(
            f"Capacité totale insuffisante : {total_capacity} < {total_students}"
        )

    assignment, broken = solve(allowed, classes_df, students_df)

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


# --- Interface Streamlit ---
st.sidebar.header("Chargement du fichier")
input_file = st.sidebar.file_uploader(
    "Votre fichier .xlsx (onglets 'liste' et 'classes')", type="xlsx"
)

if not input_file:
    st.info("⬅️ Importez le fichier .xlsx pour lancer l'affectation.")
    st.stop()

file_bytes = input_file.getvalue()

try:
    result = run_assignment(file_bytes)
except Exception as e:
    st.error(f"❌ Une erreur s'est produite : {e}")
    st.stop()

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
    file_name="assignments.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
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
        file_name="assignments.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_updated",
    )
