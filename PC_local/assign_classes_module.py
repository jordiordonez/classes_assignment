"""Moteur d'affectation des élèves aux classes.

Modèle unique CP-SAT avec contraintes souples pondérées :

  • Contraintes dures  : 1 classe par élève, capacités, options (pp/por/lat via `allowed`).
  • Contraintes souples : vœux sociaux (avec/sans) et équité (genre, niveau,
    comportement, options) — chacune pénalisée et minimisée dans un objectif pondéré.

Avantages par rapport à l'ancienne relaxation itérative :
  - les vœux avec/sans pèsent réellement sur la solution finale ;
  - résultat déterministe (graine fixée) donc reproductible ;
  - une seule résolution au lieu de trois.
"""

import pandas as pd
from ortools.sat.python import cp_model


# Poids par défaut de l'objectif. Les écarts d'équité sont exprimés en pour-mille
# (0..1000), donc un vœu social cassé « coûte » son poids en pour-mille d'équité.
# Modifiable par l'appelant pour ajuster les priorités.
DEFAULT_WEIGHTS = {
    # vœux sociaux (le vœu primaire pèse plus que le secondaire)
    "avec1": 400, "avec2": 200,
    "sans1": 400, "sans2": 200,
    # équité (écart max-min de taux de remplissage, en pour-mille)
    "fill": 1, "por": 1, "lat": 1,
    "level1": 1, "level2": 1, "level3": 1,
    "comp2": 1, "comp3": 1,
    "gender": 1,
    "pap": 1,
}


# Options « structure » imposant une contrainte DURE : un élève marqué doit aller
# dans une classe proposant la même structure.
#   • `pp`  est exclusif : une classe pp ne contient QUE des élèves pp.
#   • `por`/`lat`/`tech`/`cat` sont des structures PARTAGÉES : l'élève marqué doit y
#     aller, mais la classe peut être complétée par d'autres élèves.
# `pap` n'est PAS structurant : simple caractéristique de l'élève, comptée dans les
# tableaux de bord et équilibrée comme l'équité (genre, niveau…).
SHARED_OPTIONS = ("por", "lat", "tech", "cat")


# ──────────────────────────────────────────────────────────────────────────
#  Préparation des données
# ──────────────────────────────────────────────────────────────────────────
def load_data(students_df: pd.DataFrame, classes_df: pd.DataFrame):
    """Renomme, indexe et normalise les colonnes pp/por/lat.

    Retourne (students_df, classes_df, override_map).
    """
    students_df = students_df.rename(columns={
        "Elèves à affecter": "student",
        "Niveau": "level",
        "cat i": "cat",
    }).copy()
    classes_df = classes_df.rename(columns={
        "Nom": "class_id",
        "capacité": "capacity_override",
        "cat i": "cat",
    }).copy()

    students_df["student"] = students_df["student"].astype(str)
    students_df.set_index("student", inplace=True)
    classes_df["class_id"] = classes_df["class_id"].astype(str)
    classes_df.set_index("class_id", inplace=True)

    # Options structurantes (présentes côté élèves ET côté classes).
    for col in ("pp", "por", "lat", "tech", "cat"):
        if col not in students_df.columns:
            students_df[col] = 0
        students_df[col] = pd.to_numeric(students_df[col], errors="coerce").fillna(0).astype(int)
        if col not in classes_df.columns:
            classes_df[col] = 0
        classes_df[col] = pd.to_numeric(classes_df[col], errors="coerce").fillna(0).astype(int)

    # PAP : caractéristique binaire de l'élève (équilibrée + comptée, non structurante).
    if "pap" not in students_df.columns:
        students_df["pap"] = 0
    students_df["pap"] = pd.to_numeric(students_df["pap"], errors="coerce").fillna(0).astype(int)

    classes_df["capacity_override"] = pd.to_numeric(
        classes_df.get("capacity_override", pd.Series(dtype=float)), errors="coerce"
    )
    override_map = {
        cid: int(val)
        for cid, val in classes_df["capacity_override"].items()
        if not pd.isna(val)
    }
    return students_df, classes_df, override_map


def compute_capacities(students_df: pd.DataFrame, classes_df: pd.DataFrame, override_map: dict):
    """Calcule la colonne `capacity` (réparti les PP, puis le reste uniformément)."""
    total_students = len(students_df)
    count_pp = int(students_df["pp"].sum())
    final_caps = {}

    pp_classes = [c for c, v in classes_df["pp"].items() if v == 1]
    pp_no_ov = [c for c in pp_classes if c not in override_map]
    if pp_no_ov:
        base, extra = divmod(count_pp, len(pp_no_ov))
        for idx, c in enumerate(sorted(pp_no_ov)):
            final_caps[c] = base + (1 if idx < extra else 0)
    for c, cap in override_map.items():
        final_caps[c] = cap

    assigned_pp = sum(final_caps.get(c, 0) for c in pp_classes)
    remaining = total_students - assigned_pp
    uniform_classes = [
        c for c in classes_df.index if c not in pp_classes and c not in override_map
    ]
    if uniform_classes:
        base, extra = divmod(remaining, len(uniform_classes))
        for idx, c in enumerate(sorted(uniform_classes)):
            final_caps[c] = base + (1 if idx < extra else 0)
    elif remaining != 0:
        raise ValueError(f"Impossible de répartir {remaining} élèves restants")

    classes_df["capacity"] = classes_df.index.map(final_caps)
    return classes_df


def _option_classes(classes_df: pd.DataFrame):
    """Classes proposant chaque structure partagée (por/lat/tech/cat)."""
    all_classes = classes_df.index.tolist()
    return {
        opt: [c for c in all_classes if classes_df.at[c, opt] == 1]
        for opt in SHARED_OPTIONS
    }


def build_allowed(students_df: pd.DataFrame, classes_df: pd.DataFrame):
    """Associe à chaque élève la liste des classes possibles (contraintes dures).

    Intersection des structures demandées :
      • `pp` exclusif : un élève pp ne va qu'en classe pp ; les autres élèves ne vont
        jamais en classe pp.
      • `por`/`lat`/`tech`/`cat` partagées : l'élève marqué DOIT aller dans une classe
        proposant cette structure (la classe peut être complétée par d'autres élèves).

    Un élève dont les options ne peuvent être satisfaites par AUCUNE classe reçoit une
    liste vide ; le détail est fourni par `describe_no_class` pour inviter à corriger
    les données.
    """
    all_classes = classes_df.index.tolist()
    pp_classes = [c for c in all_classes if classes_df.at[c, "pp"] == 1]
    non_pp = [c for c in all_classes if c not in pp_classes]
    opt_classes = _option_classes(classes_df)

    allowed = {}
    for s, row in students_df.iterrows():
        # pp exclusif : élève pp -> classes pp ; sinon -> classes non-pp.
        base = pp_classes if row["pp"] == 1 else non_pp
        cur = set(base)
        for opt in SHARED_OPTIONS:
            if row[opt] == 1:
                cur &= set(opt_classes[opt])
        # Ordre stable des classes.
        allowed[s] = [c for c in all_classes if c in cur]
    return allowed


def describe_no_class(students_df: pd.DataFrame, classes_df: pd.DataFrame):
    """Pour chaque élève sans classe possible, explique le conflit d'options.

    Retourne {élève: {"options": [...], "reason": "..."}} pour affichage à l'utilisateur.
    """
    opt_classes = _option_classes(classes_df)
    has_pp_class = any(classes_df["pp"] == 1)
    allowed = build_allowed(students_df, classes_df)

    problems = {}
    for s, lst in allowed.items():
        if lst:
            continue
        row = students_df.loc[s]
        opts = (["pp"] if row["pp"] == 1 else []) + [
            o for o in SHARED_OPTIONS if row[o] == 1
        ]
        missing = [o for o in SHARED_OPTIONS if row[o] == 1 and not opt_classes[o]]
        if row["pp"] == 1 and not has_pp_class:
            missing = ["pp"] + missing
        if missing:
            reason = "aucune classe ne propose : " + ", ".join(missing)
        else:
            reason = ("aucune classe ne combine toutes les structures demandées : "
                      + " + ".join(opts))
        problems[s] = {"options": opts, "reason": reason}
    return problems


# ──────────────────────────────────────────────────────────────────────────
#  Briques du modèle
# ──────────────────────────────────────────────────────────────────────────
def _social_pairs(students_df, field):
    """Liste des paires (élève, autre) valides pour un champ avec*/sans*."""
    pairs = []
    for s, row in students_df.iterrows():
        o = row.get(field)
        if pd.notna(o) and str(o) in students_df.index and str(o) != s:
            pairs.append((s, str(o)))
    return pairs


def _both_var(model, x, s, o, c):
    """Variable booléenne = 1 ssi s ET o sont tous deux dans la classe c."""
    b = model.NewBoolVar(f"both_{s}_{o}_{c}")
    model.Add(b <= x[s, c])
    model.Add(b <= x[o, c])
    model.Add(b >= x[s, c] + x[o, c] - 1)
    return b


def _together_terms(model, x, allowed, s, o):
    """Somme des `both_var` sur les classes communes (0 ou 1 ssi même classe)."""
    common = set(allowed[s]) & set(allowed[o])
    return [_both_var(model, x, s, o, c) for c in common]


def _equity_diff(model, x, students_df, classes_df, student_filter, class_group, label):
    """Écart max-min (pour-mille) du taux de remplissage sur un sous-groupe."""
    fills = []
    for c in class_group:
        cap = int(classes_df.at[c, "capacity"])
        if cap <= 0:
            continue
        cnt = model.NewIntVar(0, cap, f"{label}_cnt_{c}")
        model.Add(cnt == sum(
            x[s, c] for s in students_df.index if (s, c) in x and student_filter(s)
        ))
        pct = model.NewIntVar(0, 1000, f"{label}_pct_{c}")
        model.AddDivisionEquality(pct, cnt * 1000, cap)
        fills.append(pct)

    if len(fills) < 2:
        return None
    mn = model.NewIntVar(0, 1000, f"{label}_min")
    mx = model.NewIntVar(0, 1000, f"{label}_max")
    model.AddMinEquality(mn, fills)
    model.AddMaxEquality(mx, fills)
    diff = model.NewIntVar(0, 1000, f"{label}_diff")
    model.Add(diff == mx - mn)
    return diff


def _gender_diff(model, x, students_df, classes_df, class_group):
    """Écart maximal (pour-mille) entre % filles et % garçons sur les classes."""
    diffs = []
    for c in class_group:
        cap = int(classes_df.at[c, "capacity"])
        if cap <= 0:
            continue
        f = model.NewIntVar(0, cap, f"f_{c}")
        g = model.NewIntVar(0, cap, f"g_{c}")
        model.Add(f == sum(
            x[s, c] for s in students_df.index
            if (s, c) in x and students_df.at[s, "Genre"] == "F"
        ))
        model.Add(g == sum(
            x[s, c] for s in students_df.index
            if (s, c) in x and students_df.at[s, "Genre"] == "G"
        ))
        fp = model.NewIntVar(0, 1000, f"fp_{c}")
        gp = model.NewIntVar(0, 1000, f"gp_{c}")
        model.AddDivisionEquality(fp, f * 1000, cap)
        model.AddDivisionEquality(gp, g * 1000, cap)
        d = model.NewIntVar(0, 1000, f"fg_{c}")
        model.AddAbsEquality(d, fp - gp)
        diffs.append(d)
    if not diffs:
        return None
    mx = model.NewIntVar(0, 1000, "fg_max")
    model.AddMaxEquality(mx, diffs)
    return mx


# ──────────────────────────────────────────────────────────────────────────
#  Résolution
# ──────────────────────────────────────────────────────────────────────────
def solve(allowed, classes_df, students_df, weights=None, time_limit=30, seed=42):
    """Affecte les élèves aux classes via un modèle unique à contraintes souples.

    Retourne (assignment, broken) :
      - assignment : {élève: classe}
      - broken     : {'avec1'|'avec2'|'sans1'|'sans2': [(source, autre), ...]}
        vœux non satisfaits dans la solution retenue.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    model = cp_model.CpModel()

    # Variables de décision + « exactement une classe » par élève.
    x = {}
    for s, cls_list in allowed.items():
        for c in cls_list:
            x[s, c] = model.NewBoolVar(f"x_{s}_{c}")
        if cls_list:
            model.AddExactlyOne(x[s, c] for c in cls_list)

    # Capacités (dures).
    for c, cap in classes_df["capacity"].items():
        vars_c = [x[s, c] for s in students_df.index if (s, c) in x]
        if vars_c:
            model.Add(sum(vars_c) <= int(cap))

    obj = []

    # Vœux sociaux (souples).
    for field in ("avec1", "avec2"):
        for s, o in _social_pairs(students_df, field):
            terms = _together_terms(model, x, allowed, s, o)
            if terms:  # pénalise « pas ensemble »
                obj.append(w[field] * (1 - sum(terms)))
    for field in ("sans1", "sans2"):
        for s, o in _social_pairs(students_df, field):
            terms = _together_terms(model, x, allowed, s, o)
            if terms:  # pénalise « ensemble »
                obj.append(w[field] * sum(terms))

    # Équité (souple).
    all_classes = classes_df.index
    diffs = {
        "fill": _equity_diff(model, x, students_df, classes_df,
                             lambda s: True, all_classes, "fill"),
        "por": _equity_diff(model, x, students_df, classes_df,
                            lambda s: students_df.at[s, "por"] == 1, all_classes, "por"),
        "lat": _equity_diff(model, x, students_df, classes_df,
                            lambda s: students_df.at[s, "lat"] == 1, all_classes, "lat"),
        "level1": _equity_diff(model, x, students_df, classes_df,
                               lambda s: int(students_df.at[s, "level"]) == 1, all_classes, "lvl1"),
        "level2": _equity_diff(model, x, students_df, classes_df,
                               lambda s: int(students_df.at[s, "level"]) == 2, all_classes, "lvl2"),
        "level3": _equity_diff(model, x, students_df, classes_df,
                               lambda s: int(students_df.at[s, "level"]) == 3, all_classes, "lvl3"),
        "comp2": _equity_diff(model, x, students_df, classes_df,
                              lambda s: students_df.at[s, "Comportement"] == 2, all_classes, "cmp2"),
        "comp3": _equity_diff(model, x, students_df, classes_df,
                              lambda s: students_df.at[s, "Comportement"] == 3, all_classes, "cmp3"),
        "pap": _equity_diff(model, x, students_df, classes_df,
                            lambda s: students_df.at[s, "pap"] == 1, all_classes, "pap"),
        "gender": _gender_diff(model, x, students_df, classes_df, all_classes),
    }
    for key, diff in diffs.items():
        if diff is not None:
            obj.append(w[key] * diff)

    if obj:
        model.Minimize(sum(obj))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.random_seed = seed
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(
            "Aucune affectation possible : vérifiez capacités et options (pp/por/lat)."
        )

    assignment = {s: c for (s, c), var in x.items() if solver.Value(var)}
    broken = report_broken(assignment, students_df)
    return assignment, broken


def report_broken(assignment, students_df):
    """Recense les vœux non satisfaits dans une affectation donnée."""
    broken = {"avec1": [], "avec2": [], "sans1": [], "sans2": []}
    for s, row in students_df.iterrows():
        for fld in ("avec1", "avec2"):
            o = row.get(fld)
            if pd.notna(o) and assignment.get(s) != assignment.get(str(o)):
                broken[fld].append((s, str(o)))
        for fld in ("sans1", "sans2"):
            o = row.get(fld)
            if pd.notna(o) and assignment.get(s) == assignment.get(str(o)):
                broken[fld].append((s, str(o)))
    return broken


# Compatibilité ascendante : ancien nom de la fonction de résolution.
def solve_two_stage(allowed, classes_df, students_df):
    return solve(allowed, classes_df, students_df)
