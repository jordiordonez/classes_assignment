import io
import pandas as pd
from xlsxwriter.utility import xl_col_to_name

def generate_tableau_sheet(writer, students_df, summary):
    students_df = students_df.sort_values(by=["classe", "student"])
    tableau_data = {
        cls: list(grp["student"]) for cls, grp in students_df.groupby("classe")
    }

    max_len = max(len(lst) for lst in tableau_data.values())
    tableau_padded = {
        cls: lst + [""] * (max_len - len(lst)) for cls, lst in tableau_data.items()
    }
    tableau_df = pd.DataFrame(tableau_padded)

    tableau_df.to_excel(writer, sheet_name="Tableau", startrow=7, startcol=1, index=False)
    ws = writer.sheets["Tableau"]

    HDR, FG_LBL, FG_CH, LVL_LBL, LVL_CH, CMP_LBL, CMP_CH, PL_LBL = range(8)
    STU_START = 8
    base_row = 100

    classes = tableau_df.columns.tolist()

    for j, cl in enumerate(classes):
        col = j + 1
        col_letter = xl_col_to_name(col)

        ws.write(HDR, col, cl)
        ws.write(FG_LBL, col, "       F            G")
        ws.write(LVL_LBL, col, "   N1      N2     N3")
        ws.write(CMP_LBL, col, "   C1      C2     C3")
        por = summary.loc[cl, "POR"] if cl in summary.index else 0
        lat = summary.loc[cl, "LAT"] if cl in summary.index else 0
        ws.write(PL_LBL, col, f"POR:{int(por)} LAT:{int(lat)}")

        ws.write(base_row, col, summary.loc[cl, "Filles"] if cl in summary.index else 0)
        ws.write(base_row + 1, col, summary.loc[cl, "Garçons"] if cl in summary.index else 0)
        ws.write(base_row + 2, col, summary.loc[cl, "Niveau1"] if cl in summary.index else 0)
        ws.write(base_row + 3, col, summary.loc[cl, "Niveau2"] if cl in summary.index else 0)
        ws.write(base_row + 4, col, summary.loc[cl, "Niveau3"] if cl in summary.index else 0)
        ws.write(base_row + 5, col, summary.loc[cl, "Comp1"] if cl in summary.index else 0)
        ws.write(base_row + 6, col, summary.loc[cl, "Comp2"] if cl in summary.index else 0)
        ws.write(base_row + 7, col, summary.loc[cl, "Comp3"] if cl in summary.index else 0)

        ws.add_sparkline(FG_CH, col, {
            "range": f"{col_letter}{base_row+1}:{col_letter}{base_row+2}",
            "type": "column",
            "min": 0
        })
        ws.add_sparkline(LVL_CH, col, {
            "range": f"{col_letter}{base_row+3}:{col_letter}{base_row+5}",
            "type": "column",
            "min": 0
        })
        ws.add_sparkline(CMP_CH, col, {
            "range": f"{col_letter}{base_row+6}:{col_letter}{base_row+8}",
            "type": "column",
            "min": 0
        })

    for i in range(len(tableau_df)):
        ws.write(STU_START + i, 0, i + 1)

    ws.set_row(FG_LBL, 15)
    ws.set_row(FG_CH, 30)
    ws.set_row(LVL_LBL, 15)
    ws.set_row(LVL_CH, 30)
    ws.set_row(CMP_LBL, 15)
    ws.set_row(CMP_CH, 30)
    ws.set_column(0, 0, 5)
    ws.set_column(1, len(classes), 15)

    return tableau_df


def validate_and_update_workbook(classes_df: pd.DataFrame, students_df: pd.DataFrame):
    # Reorder columns for readability
    cols = classes_df.columns.tolist()
    if "student" in cols and "classe" in cols:
        cols.remove("classe")
        student_index = cols.index("student")
        cols.insert(student_index + 1, "classe")
        classes_df = classes_df[cols]

    summary = classes_df.groupby("classe").agg(
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

    for col in ["%N1", "%N2", "%N3", "%Filles", "%Garçons", "%C1", "%C2", "%C3"]:
        summary[col] = 0

    summary["%N1"] = (summary["Niveau1"] / summary["Total"] * 100).round(0)
    summary["%N2"] = (summary["Niveau2"] / summary["Total"] * 100).round(0)
    summary["%N3"] = (summary["Niveau3"] / summary["Total"] * 100).round(0)
    summary["%Filles"] = (summary["Filles"] / summary["Total"] * 100).round(0)
    summary["%Garçons"] = (summary["Garçons"] / summary["Total"] * 100).round(0)
    summary["%C1"] = (summary["Comp1"] / summary["Total"] * 100).round(0)
    summary["%C2"] = (summary["Comp2"] / summary["Total"] * 100).round(0)
    summary["%C3"] = (summary["Comp3"] / summary["Total"] * 100).round(0)

    assign = dict(zip(classes_df["student"], classes_df["classe"]))

    constraints_data = []
    impossibilities_data = []

    for _, row in students_df.iterrows():
        student = row["student"]
        for fld in ("avec1", "avec2"):
            other = row.get(fld)
            if pd.notna(other):
                constraints_data.append([fld, student, other])
                if assign.get(student) != assign.get(other):
                    impossibilities_data.append([fld, student, other])
        for fld in ("sans1", "sans2"):
            other = row.get(fld)
            if pd.notna(other):
                constraints_data.append([fld, student, other])
                if assign.get(student) == assign.get(other):
                    impossibilities_data.append([fld, student, other])

    constraints_df = pd.DataFrame(constraints_data, columns=["Type", "Source", "Other"]).sort_values(by=["Type","Source"])
    impossibilites_df = pd.DataFrame(impossibilities_data, columns=["Type", "Source", "Other"]).sort_values(by=["Type","Source"])
    classes_df = classes_df.sort_values(by=["classe", "student"])
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:

        classes_df.to_excel(writer, sheet_name="Classes", index=False)
        constraints_df.to_excel(writer, sheet_name="Contraintes", index=False)
        impossibilites_df.to_excel(writer, sheet_name="Impossibilites", index=False)
        summary.to_excel(writer, sheet_name="Dashboards")
        generate_tableau_sheet(writer, classes_df, summary)
    buffer.seek(0)

    return {
        "contraintes": constraints_df,
        "impossibilites": impossibilites_df,
        "tableau_summary": summary,
        "buffer": buffer
    }
