"""Re-validation et régénération du classeur après édition manuelle des classes."""

from report_module import (
    reorder_columns,
    build_summary,
    compute_constraints,
    build_workbook,
)


def validate_and_update_workbook(classes_df, students_df):
    """Recalcule contraintes, statistiques et fichier Excel après édition manuelle.

    `classes_df` : tableau édité (colonnes student, classe, level, por, lat, Genre…).
    `students_df` : données brutes (nom + vœux avec/sans), pour revérifier les vœux.
    """
    classes_df = reorder_columns(classes_df)
    summary = build_summary(classes_df)

    assign = dict(zip(classes_df["student"].astype(str), classes_df["classe"]))
    constraints_df, impossibilites_df = compute_constraints(students_df, assign)

    buffer = build_workbook(classes_df, constraints_df, impossibilites_df, summary)
    return {
        "contraintes": constraints_df,
        "impossibilites": impossibilites_df,
        "tableau_summary": summary,
        "buffer": buffer,
    }
