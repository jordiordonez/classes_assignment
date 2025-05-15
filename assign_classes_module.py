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

import pandas as pd
import random
from ortools.sat.python import cp_model

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
        student_filter=lambda s: students_df.at[s, 'level'] == level_val,
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
    """
    Exactly the same relaxation + imbalance logic as in
    assign_classes_with_override.py
    """
    # initial pairs
    enemy1  = [(s, r['sans1']) for s,r in students_df.iterrows() if pd.notna(r['sans1'])]
    enemy2  = [(s, r['sans2']) for s,r in students_df.iterrows() if pd.notna(r['sans2'])]
    friend1 = [(s, r['avec1']) for s,r in students_df.iterrows() if pd.notna(r['avec1'])]
    friend2 = [(s, r['avec2']) for s,r in students_df.iterrows() if pd.notna(r['avec2'])]
    relax_order = [
        ('avec2', friend2, add_friend_constraints),
        ('avec1', friend1, add_friend_constraints),
        ('sans2', enemy2,   add_enemy_constraints),
        ('sans1', enemy1,   add_enemy_constraints),
    ]

    # class-groups for equity
    pp_classes     = [c for c,v in classes_df['pp'].items()  if v==1]
    por_classes    = [c for c,v in classes_df['por'].items() if v==1]
    lat_classes    = [c for c,v in classes_df['lat'].items() if v==1]
    equity_classes = [c for c in classes_df.index if c not in pp_classes]

    def build_base_model():
        x = {}
        m = cp_model.CpModel()
        # 1 élève→1 classe + capacités + ennemis hard
        for s, cls_list in allowed.items():
            for c in cls_list:
                x[s,c] = m.NewBoolVar(f"x_{s}_{c}")
            m.Add(sum(x[s,c] for c in cls_list) == 1)
        for c, cap in classes_df['capacity'].items():
            m.Add(sum(x[s2,c2] for (s2,c2) in x if c2==c) <= cap)
        add_enemy_constraints(m, x, allowed, students_df, 'sans1')
        add_enemy_constraints(m, x, allowed, students_df, 'sans2')
        return m, x

    # relaxation loop
    applied     = {nm: ps.copy() for nm,ps,_ in relax_order}
    unsatisfied = {nm: []     for nm,_,_ in relax_order}
    solver      = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60

    for name, _, fn in relax_order:
        while True:
            m, x = build_base_model()
            # reapply previous levels
            for lvl, ps, prev_fn in relax_order:
                if lvl==name: break
                for s, o in applied[lvl]:
                    prev_fn(m, x, allowed, students_df, lvl)
            # apply this level
            for s, o in applied[name]:
                fn(m, x, allowed, students_df, name)
            if solver.Solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
            if not applied[name]:
                break
            rem = random.choice(applied[name])
            applied[name].remove(rem)
            unsatisfied[name].append(rem)

    # final build + all retained constraints
    m, x = build_base_model()
    for nm, ps, fn in relax_order:
        for s, o in applied[nm]:
            fn(m, x, allowed, students_df, nm)

    # insert imbalance + equity objectives:
    fills = {}
    for c in equity_classes:
        cap = classes_df.at[c, 'capacity']
        cnt = m.NewIntVar(0, cap, f"count_{c}")
        m.Add(cnt == sum(x[s,c] for s in students_df.index if (s,c) in x))
        pct = m.NewIntVar(0, 1000, f"pct_{c}")
        m.AddDivisionEquality(pct, cnt * 1000, cap)
        fills[c] = pct

    mn = m.NewIntVar(0,1000,"min_fill");  mx = m.NewIntVar(0,1000,"max_fill")
    m.AddMinEquality(mn, list(fills.values()))
    m.AddMaxEquality(mx, list(fills.values()))
    imb = m.NewIntVar(0,1000,"imbalance");  m.Add(imb == mx - mn)

    lat_d = add_equity_constraint(m, x, students_df, classes_df,
                                 lambda s: students_df.at[s,'lat']==1, lat_classes, 'lat')
    por_d = add_equity_constraint(m, x, students_df, classes_df,
                                 lambda s: students_df.at[s,'por']==1, por_classes, 'por')
    n2_d  = add_level_equity_constraint(m, x, students_df, classes_df, 2, 'niv2')
    n3_d  = add_level_equity_constraint(m, x, students_df, classes_df, 3, 'niv3')
    fg_d  = add_gender_balance_constraint(m, x, students_df, classes_df, classes_df.index)
    b3_d  = add_behavior_equity_constraint(m, x, students_df, classes_df, 3, 'niv3')
    b2_d  = add_behavior_equity_constraint(m, x, students_df, classes_df, 2, 'niv2')

    m.Minimize(lat_d + por_d + imb + n2_d + n3_d + fg_d + b3_d + b2_d)

    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Échec même après relaxation complète")

    # extract
    assign = {s:c for (s,c),v in x.items() if solver.Value(v)}
    broken = {k:[] for k in ('avec1','avec2','sans1','sans2')}
    for s, row in students_df.iterrows():
        for col in ('avec1','avec2'):
            o = row.get(col)
            if pd.notna(o) and assign[s]!=assign.get(o):
                broken[col].append((s,o))
        for col in ('sans1','sans2'):
            o = row.get(col)
            if pd.notna(o) and assign[s]==assign.get(o):
                broken[col].append((s,o))

    return assign, broken
