import streamlit as st
import pandas as pd
import io
from xlsxwriter.utility import xl_col_to_name

from assign_classes_module import (
    load_data,
    compute_capacities,
    build_allowed,
    solve_hierarchical,
)

# --- Interface Streamlit ---
st.sidebar.header("Chargement du fichier")
input_file = st.sidebar.file_uploader(
    "Votre fichier .xlsx (onglets 'liste' et 'classes')",
    type="xlsx"
)

if input_file:
    try:
        # 1) Lecture des deux onglets
        df_students = pd.read_excel(input_file, sheet_name="liste")
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
        assignment, broken = solve_hierarchical(allowed, classes_df, students_df)

        # 3) Construction du DataFrame résultat
        students = students_df.copy()
        students['classe'] = [assignment[s] for s in students.index]
        students.reset_index(inplace=True)
        students_sorted = students.sort_values(['classe', 'student'])

        # Show basic results
        st.write("### Résultats de l'affectation")
        st.dataframe(students_sorted)

        # 4) Préparation des tableaux pour Excel
        # Calculate satisfied constraints
        all_constraints = {
            'avec1': [(s, row['avec1']) for s, row in students_df.iterrows() if pd.notna(row['avec1'])],
            'avec2': [(s, row['avec2']) for s, row in students_df.iterrows() if pd.notna(row['avec2'])],
            'sans1': [(s, row['sans1']) for s, row in students_df.iterrows() if pd.notna(row['sans1'])],
            'sans2': [(s, row['sans2']) for s, row in students_df.iterrows() if pd.notna(row['sans2'])],
        }

        satisfied = {}
        for key, pairs in all_constraints.items():
            broken_set = set(tuple(sorted(p)) for p in broken.get(key, []))
            satisfied_set = set(tuple(sorted(p)) for p in pairs) - broken_set
            satisfied[key] = list(satisfied_set)

        # Summary statistics
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
                ws.write(FG_LBL, col, "F      G")
                ws.write(LVL_LBL, col, "N1  N2  N3")
                ws.write(CMP_LBL, col, "C1  C2  C3")
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

            # Dashboards sheet
            summary.to_excel(writer, sheet_name='Dashboards')

        buffer.seek(0)

        # Download button
        st.download_button(
            "📥 Télécharger le fichier Excel",
            data=buffer,
            file_name="assignments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Une erreur s'est produite: {str(e)}")
        st.stop()

else:
    st.info("⬅️ Importez le fichier .xlsx pour lancer l'affectation.")
