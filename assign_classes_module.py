import pandas as pd
from ortools.sat.python import cp_model
import random


def load_data(students_df: pd.DataFrame, classes_df: pd.DataFrame):
    """
    Prepares and annotates the input DataFrames:
      - Renames columns
      - Indexes by student and class IDs
      - Ensures pp/por/lat flags exist
      - Applies capacity overrides
    Returns three objects: processed students_df, classes_df, override_map
    """
    # 1. Renommages
    students_df = students_df.rename(columns={
        'Elèves à affecter': 'student',
        'Niveau': 'level'
    }).copy()
    classes_df  = classes_df.rename(columns={
        'Nom': 'class_id',
        'capacité': 'capacity_override'
    }).copy()

    # 2. Indexation
    students_df['student'] = students_df['student'].astype(str)
    students_df.set_index('student', inplace=True)
    classes_df['class_id'] = classes_df['class_id'].astype(str)
    classes_df.set_index('class_id', inplace=True)

    # 3. Flags PP/POR/LAT
    for col in ['pp', 'por', 'lat']:
        if col not in students_df.columns:
            students_df[col] = 0
        students_df[col] = students_df[col].fillna(0).astype(int)

        if col not in classes_df.columns:
            classes_df[col] = 0
        classes_df[col] = classes_df[col].fillna(0).astype(int)

    # 4. Overrides
    classes_df['capacity_override'] = pd.to_numeric(
        classes_df.get('capacity_override', pd.Series()), errors='coerce'
    )
    override_map = {cid: int(val) for cid,val in classes_df['capacity_override'].items() if not pd.isna(val)}

    return students_df, classes_df, override_map

def compute_capacities(students_df: pd.DataFrame, classes_df: pd.DataFrame, override_map: dict):
    """
    Computes final class capacities based on pp counts and overrides.
    Returns the classes_df with a new 'capacity' column.
    """
    total_students = len(students_df)
    count_pp = int(students_df['pp'].sum())
    final_caps = {}

    # 1) PP distribution among PP classes without override
    pp_classes = [c for c,v in classes_df['pp'].items() if v==1]
    pp_no_ov = [c for c in pp_classes if c not in override_map]
    if pp_no_ov:
        base = count_pp // len(pp_no_ov)
        extra = count_pp % len(pp_no_ov)
        for idx,c in enumerate(sorted(pp_no_ov)):
            final_caps[c] = base + (1 if idx<extra else 0)
    # apply overrides
    for c,cap in override_map.items():
        final_caps[c] = cap

    # 2) Uniform distribution for the rest
    assigned_pp = sum(final_caps.get(c,0) for c in pp_classes)
    remaining = total_students - assigned_pp
    uniform_classes = [c for c in classes_df.index if c not in pp_classes and c not in override_map]
    if uniform_classes:
        base = remaining // len(uniform_classes)
        extra = remaining % len(uniform_classes)
        for idx,c in enumerate(sorted(uniform_classes)):
            final_caps[c] = base + (1 if idx<extra else 0)
    elif remaining !=0:
        raise ValueError(f"Impossible de répartir {remaining} élèves restants")

    classes_df['capacity'] = classes_df.index.map(final_caps)
    return classes_df

def build_allowed(students_df: pd.DataFrame, classes_df: pd.DataFrame):
    """
    Builds the 'allowed' dict mapping each student to their possible class list,
    taking into account pp/por/lat flags and fallbacks if no classes exist.
    """
    all_classes = classes_df.index.tolist()
    pp_classes  = [c for c,v in classes_df['pp'].items()  if v==1] 
    por_classes = [c for c,v in classes_df['por'].items() if v==1] 
    lat_classes = [c for c,v in classes_df['lat'].items() if v==1]
    dual_pp_por = [c for c in pp_classes if c in por_classes]
    non_pp      = [c for c in classes_df.index if c not in pp_classes]
    if not pp_classes:          # aucune classe PP déclarée
        pp_classes = non_pp     # les PP se comportent comme les autres
    if not por_classes:
        por_classes = all_classes
    if not lat_classes:
        lat_classes = all_classes

    allowed = {}
    for s,row in students_df.iterrows():
        is_pp  = row['pp']==1
        is_por = row['por']==1
        is_lat = row['lat']==1
        if is_pp and is_por:
            allowed[s] = dual_pp_por
        elif is_pp:
            allowed[s] = pp_classes
        elif is_por:
            allowed[s] = [c for c in por_classes if c not in dual_pp_por]
        elif is_lat:
            allowed[s] = lat_classes
        else:
            allowed[s] = non_pp
    return allowed

def add_enemy_constraints(model, x, allowed, students_df, field):
    """Hard enemy constraint: never same class."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] + x[other, c] <= 1)

def add_friend_constraints(model, x, allowed, students_df, field):
    """Hard friend constraint: same class."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] == x[other, c])

def add_equity_constraint(model, x, students_df, classes_df, student_filter, class_group, label):
    """
    Equity over a subgroup: minimize max–min fill % across class_group.
    Returns the IntVar holding the difference.
    """
    fill = {}
    for c in class_group:
        cap = classes_df.at[c, 'capacity']
        count = model.NewIntVar(0, cap, f'{label}_count_{c}')
        model.Add(
            count == sum(
                x[s, c]
                for s in students_df.index
                if (s, c) in x and student_filter(s)
            )
        )
        pct = model.NewIntVar(0, 1000, f'{label}_fill_{c}')
        model.AddDivisionEquality(pct, count * 1000, cap)
        fill[c] = pct

    min_fill = model.NewIntVar(0, 1000, f'{label}_min')
    max_fill = model.NewIntVar(0, 1000, f'{label}_max')
    model.AddMinEquality(min_fill, list(fill.values()))
    model.AddMaxEquality(max_fill, list(fill.values()))

    diff = model.NewIntVar(0, 1000, f'{label}_diff')
    model.Add(diff == max_fill - min_fill)
    return diff

def add_level_equity_constraint(model, x, students_df, classes_df, level_val, label):
    """Equity for students of a given level."""
    return add_equity_constraint(
        model, x, students_df, classes_df,
        student_filter=lambda s: int(students_df.at[s, 'level']) == level_val,
        class_group=classes_df.index,
        label=label
    )

def add_behavior_equity_constraint(model, x, students_df, classes_df, level_val, label):
    """Equity for students of a given behavior."""
    return add_equity_constraint(
        model, x, students_df, classes_df,
        student_filter=lambda s: students_df.at[s, 'Comportement'] == level_val,
        class_group=classes_df.index,
        label=label
    )

def add_gender_balance_constraint(model, x, students_df, classes_df, class_group):
    """Minimize gender % imbalance across class_group."""
    diffs = []
    for c in class_group:
        cap = classes_df.at[c, 'capacity']
        f_count = model.NewIntVar(0, cap, f'f_{c}')
        g_count = model.NewIntVar(0, cap, f'g_{c}')
        model.Add(f_count == sum(
            x[s, c]
            for s in students_df.index
            if (s, c) in x and students_df.at[s, 'Genre'] == 'F'
        ))
        model.Add(g_count == sum(
            x[s, c]
            for s in students_df.index
            if (s, c) in x and students_df.at[s, 'Genre'] == 'G'
        ))
        f_pct = model.NewIntVar(0, 1000, f'f_pct_{c}')
        g_pct = model.NewIntVar(0, 1000, f'g_pct_{c}')
        model.AddDivisionEquality(f_pct, f_count * 1000, cap)
        model.AddDivisionEquality(g_pct, g_count * 1000, cap)
        diff = model.NewIntVar(0, 1000, f'diff_{c}')
        model.AddAbsEquality(diff, f_pct - g_pct)
        diffs.append(diff)

    mx = model.NewIntVar(0, 1000, 'fg_max_diff')
    model.AddMaxEquality(mx, diffs)
    return mx

def solve_hierarchical(allowed, classes_df, students_df):
    """Assign students with progressive relaxation then equity optimisation.

    Phase 1 – Relax friend/enemy pairs (avec2→avec1→sans2→sans1) until a *feasible* assignment is found.
    Phase 2 – Starting from that assignment, minimise a sum of equity gaps:
        • filling imbalance (max‑min %) across non‑PP classes
        • POR / LAT distribution gap
        • level 2 & 3 distribution gap
        • behaviour 2 & 3 distribution gap
        • gender imbalance gap
    If Phase 2 is infeasible, the feasible assignment from Phase 1 is returned.
    Broken friend/enemy constraints are reported.
    """
    import random, pandas as pd
    from ortools.sat.python import cp_model

    # ---------------------------------------------------------------------------
    #  helper‑builders (safe versions ignore absent keys)
    # ---------------------------------------------------------------------------
    def add_enemy(model, x, allowed, df, field):
        for s, row in df.iterrows():
            o = row.get(field)
            if pd.notna(o) and o in df.index:
                common = set(allowed[s]) & set(allowed[o])
                for c in common:
                    if (s, c) in x and (o, c) in x:
                        model.Add(x[s, c] + x[o, c] <= 1)

    def add_friend(model, x, allowed, df, field):
        for s, row in df.iterrows():
            o = row.get(field)
            if pd.notna(o) and o in df.index:
                common = set(allowed[s]) & set(allowed[o])
                for c in common:
                    if (s, c) in x and (o, c) in x:
                        model.Add(x[s, c] == x[o, c])

    # ---------------------------------------------------------------------------
    #  pair lists & relaxation order
    # ---------------------------------------------------------------------------
    pairs = {
        'avec2': [(s, r['avec2']) for s, r in students_df.iterrows() if pd.notna(r['avec2'])],
        'avec1': [(s, r['avec1']) for s, r in students_df.iterrows() if pd.notna(r['avec1'])],
        'sans2': [(s, r['sans2']) for s, r in students_df.iterrows() if pd.notna(r['sans2'])],
        'sans1': [(s, r['sans1']) for s, r in students_df.iterrows() if pd.notna(r['sans1'])],
    }
    relax_order = [
        ('avec2', pairs['avec2'], add_friend),
        ('avec1', pairs['avec1'], add_friend),
        ('sans2', pairs['sans2'], add_enemy),
        ('sans1', pairs['sans1'], add_enemy),
    ]
    applied     = {k: v.copy() for k, v, _ in relax_order}
    unsatisfied = {k: []       for k, _, _ in relax_order}

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15

    # ---------------------------------------------------------------------------
    #  base model builder (capacity + 1‑of)
    # ---------------------------------------------------------------------------
    def build_model():
        m, x = cp_model.CpModel(), {}
        for s, cls_list in allowed.items():
            for c in cls_list:
                x[s, c] = m.NewBoolVar(f"x_{s}_{c}")
            m.Add(sum(x[s, c] for c in cls_list) == 1)
        for cls_id, cap in classes_df['capacity'].items():
            m.Add(sum(v for (stu, cls), v in x.items() if cls == cls_id) <= cap)
        return m, x

    # ---------------------------------------------------------------------------
    #  Phase 1 – progressive relaxation
    # ---------------------------------------------------------------------------
    feasible_assign = None
    while True:
        m, x = build_model()
        for nm, ps, fn in relax_order:
            for s, o in applied[nm]:
                fn(m, x, allowed, students_df, nm)
        status = solver.Solve(m)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            feasible_assign = {s: c for (s, c), var in x.items() if solver.Value(var)}
            break
        # relax one pair (priority order)
        relaxed = False
        for nm, _, _ in relax_order:
            if applied[nm]:
                rem = random.choice(applied[nm])
                applied[nm].remove(rem); unsatisfied[nm].append(rem)
                print(f"⚠️ Relaxed {nm}: {rem}")
                relaxed = True; break
        if not relaxed:
            raise RuntimeError("Aucune solution même après relaxation complète")

    # ---------------------------------------------------------------------------
    #  Phase 2 – equity optimisation (best‑effort)
    # ---------------------------------------------------------------------------
    assign = feasible_assign  # default
    try:
        m2, x2 = build_model()
        for nm, ps, fn in relax_order:
            for s, o in applied[nm]:
                fn(m2, x2, allowed, students_df, nm)

        # class groups
        pp_classes  = [c for c, v in classes_df['pp'].items()  if v == 1]
        por_classes = [c for c, v in classes_df['por'].items() if v == 1]
        lat_classes = [c for c, v in classes_df['lat'].items() if v == 1]
        equity_classes = [c for c in classes_df.index if c not in pp_classes]

        # imbalance across equity_classes
        fills = {}
        for c in equity_classes:
            cap = classes_df.at[c, 'capacity']
            cnt = m2.NewIntVar(0, cap, f'cnt_{c}')
            m2.Add(cnt == sum(x2[s, c] for s in students_df.index if (s, c) in x2))
            pct = m2.NewIntVar(0, 1000, f'pct_{c}')
            m2.AddDivisionEquality(pct, cnt * 1000, cap)
            fills[c] = pct
        mn = m2.NewIntVar(0, 1000, 'mn'); mx = m2.NewIntVar(0, 1000, 'mx')
        m2.AddMinEquality(mn, list(fills.values()))
        m2.AddMaxEquality(mx, list(fills.values()))
        imb = m2.NewIntVar(0, 1000, 'imb'); m2.Add(imb == mx - mn)

        # helper wrappers already defined in module scope
        from assign_classes_module import (
            add_equity_constraint, add_level_equity_constraint,
            add_behavior_equity_constraint, add_gender_balance_constraint
        )
        lat_d = add_equity_constraint(  m2, x2, students_df, classes_df,
                                        lambda s: students_df.at[s, 'lat'] == 1,
                                        lat_classes, 'lat')
        por_d = add_equity_constraint(  m2, x2, students_df, classes_df,
                                        lambda s: students_df.at[s, 'por'] == 1,
                                        por_classes, 'por')
        n2_d  = add_level_equity_constraint(m2, x2, students_df, classes_df, 2, 'niv2')
        n3_d  = add_level_equity_constraint(m2, x2, students_df, classes_df, 3, 'niv3')
        fg_d  = add_gender_balance_constraint(m2, x2, students_df, classes_df, classes_df.index)
        b3_d  = add_behavior_equity_constraint(m2, x2, students_df, classes_df, 3, 'comp3')
        b2_d  = add_behavior_equity_constraint(m2, x2, students_df, classes_df, 2, 'comp2')

        m2.Minimize(lat_d + por_d + imb + n2_d + n3_d + fg_d + b3_d + b2_d)

        if solver.Solve(m2) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assign = {s: c for (s, c), var in x2.items() if solver.Value(var)}
        else:
            print("⚠️ Optimisation équité infaisable – conservation de la solution faisable.")
    except Exception as e:
        print(f"⚠️ Optimisation équité échouée ({e}) – conservation de la solution faisable.")

    # ---------------------------------------------------------------------------
    #  Broken constraints report
    # ---------------------------------------------------------------------------
    broken = {k: [] for k in ('avec1', 'avec2', 'sans1', 'sans2')}
    for s, row in students_df.iterrows():
        for fld in ('avec1', 'avec2'):
            o = row.get(fld)
            if pd.notna(o) and assign.get(s) != assign.get(o):
                broken[fld].append((s, o))
        for fld in ('sans1', 'sans2'):
            o = row.get(fld)
            if pd.notna(o) and assign.get(s) == assign.get(o):
                broken[fld].append((s, o))

    return assign, broken


# ────────────────────────────────────────────────────────────────
#  Affectation en deux étapes : d’abord les PP, puis les autres
# ────────────────────────────────────────────────────────────────
def solve_two_stage(allowed, classes_df, students_df):
    """
    1) Place les PP dans les classes PP (avec relaxation locale).
    2) Place les autres élèves en tenant compte des capacités résiduelles.
    3) Relance une optimisation « équité globale » en bloquant la place des PP.
    4) PP only  • 2) reste du monde  • 3) équité globale.
    """
    # ─────────── groupes de classes utilisés PARTOUT ────────────
    pp_classes  = [c for c, v in classes_df['pp' ].items() if v == 1]
    por_classes = [c for c, v in classes_df['por'].items() if v == 1]
    lat_classes = [c for c, v in classes_df['lat'].items() if v == 1]
    # --- groupes --------------------------------------------------------------
    pp_students   = students_df.index[students_df['pp'] == 1].tolist()
    non_pp_students = [s for s in students_df.index if s not in pp_students]
    pp_classes    = [c for c, v in classes_df['pp'].items() if v == 1]

    # ---------- A) affectation PP --------------------------------------------
    if pp_classes:        # seulement si des classes PP existent
        # juste avant l’appel à solve_hierarchical pour les PP
        only_pp_pairs = lambda lst: [(a,b) for a,b in lst if a in pp_students and b in pp_students]

        allowed_pp = {s: [c for c in allowed[s] if c in pp_classes] for s in pp_students}
        assign_pp, _ = solve_hierarchical(
            allowed_pp,
            classes_df.loc[pp_classes].copy(),
            students_df.loc[pp_students].copy()
)

    else:
        assign_pp = {}

    # ---------- B) capacités résiduelles -------------------------------------
    resid_cap = classes_df['capacity'].copy()
    for cls in assign_pp.values():
        resid_cap[cls] -= 1
    classes_df_resid = classes_df.copy()
    classes_df_resid['capacity'] = resid_cap

    # ---------- C) affectation reste du monde ---------------------------------
    allowed_rest = {s: allowed[s] for s in non_pp_students}
    assign_rest, broken_rest = solve_hierarchical(
        allowed_rest,
        classes_df_resid,
        students_df.loc[non_pp_students].copy()
    )

    # ---------- D) optimisation globale (soft-constraints) -------------------
    # On repart avec TOUTES les variables, MAIS on fige les PP.
    # build_model_global == build_model() mais avec un fix pour les PP
    from ortools.sat.python import cp_model
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 30

    def build_model_global():
        m, x = cp_model.CpModel(), {}
        for s, cls_list in allowed.items():
            for c in cls_list:
                x[s, c] = m.NewBoolVar(f"x_{s}_{c}")
            m.Add(sum(x[s, c] for c in cls_list) == 1)

        for c, cap in classes_df['capacity'].items():
            m.Add(sum(v for (stu, cls), v in x.items() if cls == c) <= cap)

        # ▶️ on gèle les PP
        for s, cls in assign_pp.items():
            m.Add(x[s, cls] == 1)

        return m, x

    try:
        m, x = build_model_global()
        # on réinjecte TOUTES les contraintes non relaxées
        for nm, lst, fn in [('avec2', [], add_friend_constraints),
                            ('avec1', [], add_friend_constraints),
                            ('sans2', [], add_enemy_constraints),
                            ('sans1', [], add_enemy_constraints)]:
            for s, o in lst:
                fn(m, x, allowed, students_df, nm)

        # ⚖️ objectifs d’équité sur **toutes** les classes
        pp_classes_flag = [c for c, v in classes_df['pp'].items() if v == 1]
        equity_classes  = classes_df.index      # inclut les PP !

        # ré-utilise les helpers déjà définis
        imb = add_equity_constraint(
            m, x, students_df, classes_df,
            student_filter=lambda s: True,
            class_group=equity_classes,
            label='fill'
        )
        por_d = add_equity_constraint(
            m, x, students_df, classes_df,
            student_filter=lambda s: students_df.at[s,'por'] == 1,
            class_group=por_classes or classes_df.index,
            label='por'
        )
        lat_d = add_equity_constraint(
            m, x, students_df, classes_df,
            student_filter=lambda s: students_df.at[s,'lat'] == 1,
            class_group=lat_classes or classes_df.index,
            label='lat'
        )
        n2_d = add_level_equity_constraint(m, x, students_df, classes_df, 2, 'niv2')
        n3_d = add_level_equity_constraint(m, x, students_df, classes_df, 3, 'niv3')
        fg_d = add_gender_balance_constraint(m, x, students_df, classes_df, classes_df.index)
        b2_d = add_behavior_equity_constraint(m, x, students_df, classes_df, 2, 'comp2')
        b3_d = add_behavior_equity_constraint(m, x, students_df, classes_df, 3, 'comp3')

        m.Minimize(imb + por_d + lat_d + n2_d + n3_d + fg_d + b2_d + b3_d)

        if solver.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            assign = {s: c for (s, c), v in x.items() if solver.Value(v)}
        else:
            print("⚠️ Optimisation finale infaisable, on garde la solution précédente")
            assign = {**assign_pp, **assign_rest}

    except Exception as e:
        print("⚠️ Équité globale non exécutée :", e)
        assign = {**assign_pp, **assign_rest}

    # ---------- E) contraintes cassées ----------------------------------------
    broken = {'avec1': [], 'avec2': [], 'sans1': [], 'sans2': []}
    for s, r in students_df.iterrows():
        for fld in ('avec1', 'avec2'):
            o = r.get(fld)
            if pd.notna(o) and assign.get(s) != assign.get(o):
                broken[fld].append((s, o))
        for fld in ('sans1', 'sans2'):
            o = r.get(fld)
            if pd.notna(o) and assign.get(s) == assign.get(o):
                broken[fld].append((s, o))

    return assign, broken
