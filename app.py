import streamlit as st
import pandas as pd
import io
from xlsxwriter.utility import xl_col_to_name

from update_classes_module import validate_and_update_workbook


from assign_classes_module import (
    load_data,
    compute_capacities,
    build_allowed,
    solve_hierarchical,
    solve_two_stage
)
import streamlit as st

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
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("""
## 📤 Structure du fichier Excel de sortie

Une fois le traitement terminé, l’application génère un fichier Excel comportant plusieurs feuilles :

---

### 🏫 Feuille `Classes` : Affectations finales

Contient la liste complète des élèves avec leur affectation.

| Colonne        | Description                            |
|----------------|----------------------------------------|
| `student`      | Nom de l’élève                         |
| `Genre`        | F ou G                                 |
| `por`, `lat`, `pp` | Options choisies (0 ou 1)          |
| `level`        | Niveau scolaire                        |
| `Comportement` | Indice de priorité (1 = prioritaire)  |
| `avec1`, `avec2`, `sans1`, `sans2` | Souhaits sociaux    |
| `classe`       | Classe attribuée à l’élève             |

---

### ❗ Feuille `Impossibilites` : Contraintes non résolues

Liste des contraintes non respectées (souhaits impossibles à satisfaire).

| Colonne  | Description                      |
|----------|----------------------------------|
| `Type`   | Type de souhait (ex : `avec1`)   |
| `Source` | Élève à l’origine du souhait     |
| `Other`  | Élève concerné par le conflit    |

---

### ⚠️ Feuille `Contraintes` : Contraintes prises en compte

Contient les contraintes traitées.

| Colonne  | Description                      |
|----------|----------------------------------|
| `Type`   | Type de contrainte (ex : `avec2`)|
| `Source` | Élève émetteur                   |
| `Other`  | Élève ciblé                      |

---

### 📊 Feuille `Tableau` : Matrice de répartition

Structure tabulaire croisant les genres (F/G), niveaux (N1, N2, N3) et classes (A, B, C...).

> Utile pour visualiser les équilibres par classe, niveau et genre.

---

### 📈 Feuille `Dashboards` : Statistiques globales

Tableau de bord synthétique par classe :

| Colonne      | Description                                |
|--------------|--------------------------------------------|
| `classe`     | Nom de la classe                           |
| `Total`      | Nombre total d’élèves                      |
| `Niveau1-3`  | Répartition par niveau                     |
| `POR`, `LAT` | Nombre d’élèves par option                 |
| `Filles`, `Garçons` | Répartition par genre             |
| `Comp1-3`    | Niveaux de comportement                    |
| `%N1-N3`, `%Filles`, `%Garçons`, `%C1-C3` | Pourcentages |

---
""")

# --- Interface Streamlit ---
st.sidebar.header("Chargement du fichier")
input_file = st.sidebar.file_uploader(
    "Votre fichier .xlsx (onglets 'liste' et 'classes')",
    type="xlsx"
)
if input_file:
    if "students_sorted" not in st.session_state:
        try:
            # 1) Lecture des deux onglets
            df_students = pd.read_excel(input_file, sheet_name="liste")
            # Normalize column names
            df_students = df_students.rename(columns={"Elèves à affecter": "student"})

            df_classes = pd.read_excel(input_file, sheet_name="classes")

            # 2) Vérifications préliminaires
            total_students = len(df_students)
            
            # Define equity classes before processing
            equity_classes = [
                c for c in df_classes.index
                if df_classes.at[c, 'pp'] == 0  # Include all non-PP
            ]

            # Define class groups
            pp_classes = [c for c, v in df_classes['pp'].items() if v == 1]
            por_classes = [c for c, v in df_classes['por'].items() if v == 1]
            lat_classes = [c for c, v in df_classes['lat'].items() if v == 1]
            print(f"Classes PP: {pp_classes}, Classes POR: {por_classes}, Classes LAT: {lat_classes}")
            # Print warnings if any required class type is missing
            if not pp_classes:
                st.warning("⚠️ Aucune classe PP trouvée → tous les PP peuvent aller dans n'importe quelle classe")
            if not por_classes:
                st.warning("⚠️ Aucune classe POR trouvée → tous les POR peuvent aller dans n'importe quelle classe")
            if not lat_classes:
                st.warning("⚠️ Aucune classe LAT trouvée → tous les LAT peuvent aller dans n'importe quelle classe")

            # 3) Logique métier
            students_df, classes_df, override_map = load_data(df_students, df_classes)
            classes_df = compute_capacities(students_df, classes_df, override_map)
            allowed = build_allowed(students_df, classes_df)

            # 4) Vérification de la capacité totale
            total_capacity = classes_df['capacity'].sum()
            st.write(f"Total students = {total_students}, total capacity = {total_capacity}")
            if total_capacity < total_students:
                raise RuntimeError(f"Capacité totale insuffisante : {total_capacity} < {total_students}")

            # 5) Vérification des affectations possibles
            for s, cls_list in allowed.items():
                if len(cls_list) == 0:
                    st.warning(f"⚠️ Élève sans affectation possible : {s}")

            # 6) Résolution
            assignment, broken = solve_two_stage(allowed, classes_df, students_df)


            # 3) Construction du DataFrame résultat
            students = students_df.copy()
            students['classe'] = [assignment[s] for s in students.index]
            students.reset_index(inplace=True)
            students_sorted = students.sort_values(['classe', 'student'])
            # Réorganiser les colonnes : mettre 'classe' juste après 'student'
            cols = students_sorted.columns.tolist()
            if "student" in cols and "classe" in cols:
                cols.remove("classe")
                insert_pos = cols.index("student") + 1
                cols.insert(insert_pos, "classe")
                students_sorted = students_sorted[cols]

            st.session_state["students_sorted"] = students_sorted
            st.session_state["students_df"] = df_students
            st.session_state["broken"] = broken


        except Exception as e:
            st.error(f"❌ Une erreur s'est produite: {str(e)}")
            st.stop()

    # Show basic results
    st.write("### Résultats de l'affectation")

    # 4) Préparation des tableaux pour Excel
    # Calculate satisfied constraints
    students_df = st.session_state["students_df"]
    broken = st.session_state["broken"]

    all_constraints = {
        'avec1': [(s, row['avec1']) for s, row in students_df.iterrows() if pd.notna(row['avec1'])],
        'avec2': [(s, row['avec2']) for s, row in students_df.iterrows() if pd.notna(row['avec2'])],
        'sans1': [(s, row['sans1']) for s, row in students_df.iterrows() if pd.notna(row['sans1'])],
        'sans2': [(s, row['sans2']) for s, row in students_df.iterrows() if pd.notna(row['sans2'])],
    }

    satisfied = {}
    for key, pairs in all_constraints.items():
        broken_set = set(tuple(sorted(map(str, p))) for p in broken.get(key, []))
        satisfied_set = set(tuple(sorted(map(str, p))) for p in pairs) - broken_set
        satisfied[key] = list(satisfied_set)

    # Summary statistics
    students_sorted = st.session_state["students_sorted"]
    summary = students_sorted.groupby('classe').agg(
        Total=('student', 'count'),
        Niveau1=('level', lambda x: (x == 1).sum()),
        Niveau2=('level', lambda x: (x == 2).sum()),
        Niveau3=('level', lambda x: (x == 3).sum()),
        POR=('por', 'sum'),
        LAT=('lat', 'sum'),
        Filles=('Genre', lambda x: (x == 'F').sum()),
        Garçons=('Genre', lambda x: (x == 'G').sum()),
        Comp1=('Comportement', lambda x: (x == 1).sum()),
        Comp2=('Comportement', lambda x: (x == 2).sum()),
        Comp3=('Comportement', lambda x: (x == 3).sum()),
    )

    # Add percentages
    summary['%N1'] = (summary['Niveau1'] / summary['Total'] * 100).round(0)
    summary['%N2'] = (summary['Niveau2'] / summary['Total'] * 100).round(0)
    summary['%N3'] = (summary['Niveau3'] / summary['Total'] * 100).round(0)
    summary['%Filles'] = (summary['Filles'] / summary['Total'] * 100).round(0)
    summary['%Garçons'] = (summary['Garçons'] / summary['Total'] * 100).round(0)
    summary['%C1'] = (summary['Comp1'] / summary['Total'] * 100).round(0)
    summary['%C2'] = (summary['Comp2'] / summary['Total'] * 100).round(0)
    summary['%C3'] = (summary['Comp3'] / summary['Total'] * 100).round(0)

    # Generate Excel file
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        # Classes sheet
        # Réorganiser les colonnes : mettre 'classe' juste après 'student'
        cols = students_sorted.columns.tolist()
        if "student" in cols and "classe" in cols:
            cols.remove("classe")
            insert_pos = cols.index("student") + 1
            cols.insert(insert_pos, "classe")
            students_sorted = students_sorted[cols]

        students_sorted.to_excel(writer, sheet_name='Classes', index=False)

        # Impossibilites sheet - broken constraints
        imposs_data = []
        for kind, pairs in broken.items():
            for s, o in pairs:
                imposs_data.append([kind, s, o])
        if imposs_data:
            pd.DataFrame(imposs_data, columns=['Type', 'Source', 'Other']).to_excel(
                writer, sheet_name='Impossibilites', index=False
            )

        # Contraintes sheet - satisfied constraints
        sat_data = []
        for kind, pairs in satisfied.items():
            for s, o in pairs:
                sat_data.append([kind, s, o])
        if sat_data:
            pd.DataFrame(sat_data, columns=['Type', 'Source', 'Other']).to_excel(
                writer, sheet_name='Contraintes', index=False
            )

        # Tableau sheet with sparklines
        tableau_data = {
            cls: list(grp['student']) for cls, grp in students_sorted.groupby('classe')
        }
        max_len = max(len(lst) for lst in tableau_data.values())
        tableau_padded = {
            cls: lst + [''] * (max_len - len(lst)) 
            for cls, lst in tableau_data.items()
        }
        tableau_df = pd.DataFrame(tableau_padded)
        
        tableau_df.to_excel(writer, sheet_name='Tableau', startrow=7, startcol=1, index=False)
        ws = writer.sheets['Tableau']

        # Headers and formatting
        HDR, FG_LBL, FG_CH, LVL_LBL, LVL_CH, CMP_LBL, CMP_CH, PL_LBL = range(8)
        STU_START = 8
        base_row = 100  # Raw data for sparklines starts here

        classes = tableau_df.columns.tolist()
        for j, cl in enumerate(classes):
            col = j + 1
            ws.write(HDR, col, cl)
            ws.write(FG_LBL, col, "       F            G")
            ws.write(LVL_LBL, col, "   N1      N2     N3")
            ws.write(CMP_LBL, col, "   C1      C2     C3")
            por = summary.loc[cl, 'POR']
            lat = summary.loc[cl, 'LAT']
            ws.write(PL_LBL, col, f"POR:{int(por)} LAT:{int(lat)}")

            # Write raw data for sparklines
            ws.write(base_row, col, summary.loc[cl, 'Filles'])
            ws.write(base_row + 1, col, summary.loc[cl, 'Garçons'])
            ws.write(base_row + 2, col, summary.loc[cl, 'Niveau1'])
            ws.write(base_row + 3, col, summary.loc[cl, 'Niveau2'])
            ws.write(base_row + 4, col, summary.loc[cl, 'Niveau3'])
            ws.write(base_row + 5, col, summary.loc[cl, 'Comp1'])
            ws.write(base_row + 6, col, summary.loc[cl, 'Comp2'])
            ws.write(base_row + 7, col, summary.loc[cl, 'Comp3'])

            # Add sparklines
            col_letter = xl_col_to_name(col)
            ws.add_sparkline(FG_CH, col, {
                'range': f'{col_letter}{base_row+1}:{col_letter}{base_row+2}',
                'type': 'column',
                'min': 0
            })
            ws.add_sparkline(LVL_CH, col, {
                'range': f'{col_letter}{base_row+3}:{col_letter}{base_row+5}',
                'type': 'column',
                'min': 0
            })
            ws.add_sparkline(CMP_CH, col, {
                'range': f'{col_letter}{base_row+6}:{col_letter}{base_row+8}',
                'type': 'column',
                'min': 0
            })

        # Row numbers
        for i in range(len(tableau_df)):
            ws.write(STU_START + i, 0, i + 1)

        ws.set_row(FG_LBL, 15)
        ws.set_row(FG_CH, 30)
        ws.set_row(LVL_LBL,15)
        ws.set_row(LVL_CH,30)
        ws.set_row(CMP_LBL,15)
        ws.set_row(CMP_CH,30)
        ws.set_column(0,0,5)
        ws.set_column(1,len(classes),15)

        # Dashboards sheet
        summary.to_excel(writer, sheet_name='Dashboards')

    buffer.seek(0)

    # Download button
    st.download_button(
        "📥 Télécharger le fichier Excel",
        data=buffer,
        file_name="assignments.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_initial"
    )
    


    st.markdown("""
### ✏️ Modifier les affectations manuellement

Vous pouvez modifier les classes attribuées aux élèves directement dans le tableau ci-dessous.  
Une fois vos modifications terminées, cliquez sur **🔁 Rafraîchir et vérifier les contraintes** pour :

- Mettre à jour toutes les feuilles (`Classes`, `Contraintes`, `Impossibilites`, `Tableau`, `Dashboards`)
- Voir les éventuelles contraintes non respectées
- Télécharger le nouveau fichier Excel mis à jour

""")
    edited_df = st.data_editor(
        st.session_state["students_sorted"],
        column_config={
            "classe": st.column_config.SelectboxColumn("Classe", options=st.session_state["students_sorted"]["classe"].unique().tolist())
        },
        use_container_width=True
    )


    if st.button("🔁 Rafraîchir et vérifier les contraintes"):
        results = validate_and_update_workbook(edited_df, st.session_state["students_df"])

        st.success("📦 Fichier 'assignments.xlsx' mis à jour avec succès.")

        st.download_button(
            "📥 Télécharger le fichier Excel mis à jour",
            data=results["buffer"],
            file_name="assignments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_updated"
        )


else:
    st.info("⬅️ Importez le fichier .xlsx pour lancer l'affectation.")


