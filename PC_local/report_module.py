"""Génération des feuilles de sortie (Classes, Contraintes, Impossibilites,
Tableau, Dashboards) — partagé par l'affectation initiale et la mise à jour
manuelle, afin d'éviter toute duplication."""

import io
import pandas as pd
from xlsxwriter.utility import xl_col_to_name


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Place la colonne `classe` juste après `student`."""
    cols = df.columns.tolist()
    if "student" in cols and "classe" in cols:
        cols.remove("classe")
        cols.insert(cols.index("student") + 1, "classe")
        return df[cols]
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques agrégées par classe (effectifs + pourcentages)."""
    summary = df.groupby("classe").agg(
        Total=("student", "count"),
        Niveau1=("level", lambda x: (x == 1).sum()),
        Niveau2=("level", lambda x: (x == 2).sum()),
        Niveau3=("level", lambda x: (x == 3).sum()),
        POR=("por", "sum"),
        LAT=("lat", "sum"),
        Filles=("Genre", lambda x: (x == "F").sum()),
        Garçons=("Genre", lambda x: (x == "G").sum()),
        Comp1=("Comportement", lambda x: (x == 1).sum()),
        Comp2=("Comportement", lambda x: (x == 2).sum()),
        Comp3=("Comportement", lambda x: (x == 3).sum()),
    )
    pct = {
        "%N1": "Niveau1", "%N2": "Niveau2", "%N3": "Niveau3",
        "%Filles": "Filles", "%Garçons": "Garçons",
        "%C1": "Comp1", "%C2": "Comp2", "%C3": "Comp3",
    }
    for out, src in pct.items():
        summary[out] = (summary[src] / summary["Total"] * 100).round(0)
    return summary


def compute_constraints(students_df: pd.DataFrame, assign: dict):
    """Construit les DataFrames Contraintes (tous les vœux) et Impossibilites
    (vœux non respectés), à partir des données brutes et d'une affectation.

    `students_df` doit contenir la colonne `student` (nom) et les champs
    avec1/avec2/sans1/sans2. `assign` mappe un nom d'élève vers sa classe.
    """
    constraints, impossibilities = [], []
    for _, row in students_df.iterrows():
        student = str(row["student"])
        for fld in ("avec1", "avec2"):
            other = row.get(fld)
            if pd.notna(other):
                constraints.append([fld, student, str(other)])
                if assign.get(student) != assign.get(str(other)):
                    impossibilities.append([fld, student, str(other)])
        for fld in ("sans1", "sans2"):
            other = row.get(fld)
            if pd.notna(other):
                constraints.append([fld, student, str(other)])
                if assign.get(student) == assign.get(str(other)):
                    impossibilities.append([fld, student, str(other)])

    cols = ["Type", "Source", "Other"]
    constraints_df = pd.DataFrame(constraints, columns=cols).sort_values(["Type", "Source"])
    impossibilites_df = pd.DataFrame(impossibilities, columns=cols).sort_values(["Type", "Source"])
    return constraints_df, impossibilites_df


def generate_tableau_sheet(writer, students_df: pd.DataFrame, summary: pd.DataFrame):
    """Feuille matricielle : élèves par classe + sparklines (genre/niveau/comp)."""
    students_df = students_df.sort_values(by=["classe", "student"])
    tableau_data = {cls: list(grp["student"]) for cls, grp in students_df.groupby("classe")}
    max_len = max((len(lst) for lst in tableau_data.values()), default=0)
    tableau_padded = {
        cls: lst + [""] * (max_len - len(lst)) for cls, lst in tableau_data.items()
    }
    tableau_df = pd.DataFrame(tableau_padded)

    tableau_df.to_excel(writer, sheet_name="Tableau", startrow=7, startcol=1, index=False)
    ws = writer.sheets["Tableau"]

    HDR, FG_LBL, FG_CH, LVL_LBL, LVL_CH, CMP_LBL, CMP_CH, PL_LBL = range(8)
    STU_START = 8
    base_row = 100

    def stat(cl, key):
        return summary.loc[cl, key] if cl in summary.index else 0

    classes = tableau_df.columns.tolist()
    for j, cl in enumerate(classes):
        col = j + 1
        col_letter = xl_col_to_name(col)
        ws.write(HDR, col, cl)
        ws.write(FG_LBL, col, "       F            G")
        ws.write(LVL_LBL, col, "   N1      N2     N3")
        ws.write(CMP_LBL, col, "   C1      C2     C3")
        ws.write(PL_LBL, col, f"POR:{int(stat(cl, 'POR'))} LAT:{int(stat(cl, 'LAT'))}")

        for offset, key in enumerate(
            ["Filles", "Garçons", "Niveau1", "Niveau2", "Niveau3", "Comp1", "Comp2", "Comp3"]
        ):
            ws.write(base_row + offset, col, stat(cl, key))

        ws.add_sparkline(FG_CH, col, {
            "range": f"{col_letter}{base_row + 1}:{col_letter}{base_row + 2}",
            "type": "column", "min": 0,
        })
        ws.add_sparkline(LVL_CH, col, {
            "range": f"{col_letter}{base_row + 3}:{col_letter}{base_row + 5}",
            "type": "column", "min": 0,
        })
        ws.add_sparkline(CMP_CH, col, {
            "range": f"{col_letter}{base_row + 6}:{col_letter}{base_row + 8}",
            "type": "column", "min": 0,
        })

    for i in range(len(tableau_df)):
        ws.write(STU_START + i, 0, i + 1)

    for r in (FG_LBL, LVL_LBL, CMP_LBL):
        ws.set_row(r, 15)
    for r in (FG_CH, LVL_CH, CMP_CH):
        ws.set_row(r, 30)
    ws.set_column(0, 0, 5)
    ws.set_column(1, len(classes), 15)
    return tableau_df


def build_workbook(students_sorted: pd.DataFrame, constraints_df: pd.DataFrame,
                   impossibilites_df: pd.DataFrame, summary: pd.DataFrame) -> io.BytesIO:
    """Assemble le classeur Excel complet et renvoie un buffer prêt à télécharger."""
    students_sorted = reorder_columns(students_sorted).sort_values(["classe", "student"])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        students_sorted.to_excel(writer, sheet_name="Classes", index=False)
        if not impossibilites_df.empty:
            impossibilites_df.to_excel(writer, sheet_name="Impossibilites", index=False)
        if not constraints_df.empty:
            constraints_df.to_excel(writer, sheet_name="Contraintes", index=False)
        generate_tableau_sheet(writer, students_sorted, summary)
        summary.to_excel(writer, sheet_name="Dashboards")
    buffer.seek(0)
    return buffer
