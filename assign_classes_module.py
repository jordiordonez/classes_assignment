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

def add_enemy_constraints(model, x, allowed, students_df, field):
    """Ajoute les contraintes 'sans1' / 'sans2' (ennemis ne peuvent pas se retrouver dans la même classe)."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] + x[other, c] <= 1)

def add_friend_constraints(model, x, allowed, students_df, field):
    """Ajoute les contraintes 'avec1' / 'avec2' (amis doivent être dans la même classe)."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] == x[other, c])

def build_full_model(allowed, classes_df):
    """Builds the base model with variables, 1 student → 1 class and capacity constraints."""
    x = {}
    model = cp_model.CpModel()
    
    # 1) Variables and "1 student → 1 class"
    for s, cls_list in allowed.items():
        for c in cls_list:
            x[s, c] = model.NewBoolVar(f"x_{s}_{c}")
        model.Add(sum(x[s, c] for c in cls_list) == 1)
    
    # 2) Capacity constraints
    for c, cap in classes_df['capacity'].items():
        model.Add(sum(var for (s2, c2), var in x.items() if c2 == c) <= cap)
    
    return model, x

def solve_hierarchical(allowed, classes_df, students_df):
    # 0️⃣ Define class groups like in affect2.py
    pp_classes = [c for c, v in classes_df['pp'].items() if v == 1]
    por_classes = [c for c, v in classes_df['por'].items() if v == 1]
    lat_classes = [c for c, v in classes_df['lat'].items() if v == 1]
    equity_classes = [c for c in classes_df.index if c not in pp_classes]

    # Store initial pairs for each level
    enemy1 = [(s, row['sans1']) for s,row in students_df.iterrows() if pd.notna(row['sans1'])]
    enemy2 = [(s, row['sans2']) for s,row in students_df.iterrows() if pd.notna(row['sans2'])]
    friend1 = [(s, row['avec1']) for s,row in students_df.iterrows() if pd.notna(row['avec1'])]
    friend2 = [(s, row['avec2']) for s,row in students_df.iterrows() if pd.notna(row['avec2'])]
    
    relax_order = [
        ('avec2', friend2, add_friend_constraints),
        ('avec1', friend1, add_friend_constraints),
        ('sans2', enemy2, add_enemy_constraints),
        ('sans1', enemy1, add_enemy_constraints),
    ]
    def add_equity_constraint(model, x, students_df, classes_df, student_filter, class_group, label):
        """Adds equity constraints for specific student groups across classes."""
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
        """Adds equity constraints for students of specific level."""
        return add_equity_constraint(
            model, x, students_df, classes_df,
            student_filter=lambda s: students_df.at[s, 'level'] == level_val,
            class_group=classes_df.index,
            label=label
        )
    
    def add_behavior_equity_constraint(model, x, students_df, classes_df, level_val, label):
        """Adds equity constraints for students with specific behavior level."""
        return add_equity_constraint(
            model, x, students_df, classes_df,
            student_filter=lambda s: students_df.at[s, 'Comportement'] == level_val,
            class_group=classes_df.index,
            label=label
        )
    
    def add_gender_balance_constraint(model, x, students_df, classes_df, class_group):
        """Adds gender balance constraints across classes."""
        diffs = []
        for c in class_group:
            cap = classes_df.at[c, 'capacity']
            f_count = model.NewIntVar(0, cap, f'f_{c}')
            g_count = model.NewIntVar(0, cap, f'g_{c}')
    
            model.Add(f_count == sum(x[s, c] for s in students_df.index if (s, c) in x and students_df.at[s, 'Genre'] == 'F'))
            model.Add(g_count == sum(x[s, c] for s in students_df.index if (s, c) in x and students_df.at[s, 'Genre'] == 'G'))
    
            f_pct = model.NewIntVar(0, 1000, f'f_pct_{c}')
            g_pct = model.NewIntVar(0, 1000, f'g_pct_{c}')
            model.AddDivisionEquality(f_pct, f_count * 1000, cap)
            model.AddDivisionEquality(g_pct, g_count * 1000, cap)
    
            diff = model.NewIntVar(0, 1000, f'diff_{c}')
            model.AddAbsEquality(diff, f_pct - g_pct)
            diffs.append(diff)
    
        max_diff = model.NewIntVar(0, 1000, 'fg_max_diff')
        model.AddMaxEquality(max_diff, diffs)
        return max_diff
    # Initialize applied and unsatisfied constraints
    applied = {name: pairs.copy() for name, pairs, _ in relax_order}
    unsatisfied = {name: [] for name, _, _ in relax_order}
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5  # Match affect2.py timeout

    # ... rest of the helper functions remain the same ...

    # Main optimization loop
    for name, _, adder in relax_order:
        while True:
            model, x = build_full_model(allowed, classes_df)  # Pass the parameters
            
            # Apply all validated pairs from previous levels
            for lvl, pairs, fn in relax_order:
                if lvl == name:
                    break
                for s, other in applied[lvl]:
                    fn(model, x, allowed, students_df, lvl)
            
            # Apply current level pairs
            for s, other in applied[name]:
                adder(model, x, allowed, students_df, name)

            # Add all optimization objectives
            fill_percent = {}
            for c in equity_classes:
                cap = classes_df.at[c, 'capacity']
                count = model.NewIntVar(0, cap, f'count_{c}')
                model.Add(count == sum(x[s, c] for s in students_df.index if (s, c) in x))
                scaled = model.NewIntVar(0, 1000, f'fill_{c}')
                model.AddDivisionEquality(scaled, count * 1000, cap)
                fill_percent[c] = scaled

            # Add min/max fill constraints
            min_fill = model.NewIntVar(0, 1000, 'min_fill')
            max_fill = model.NewIntVar(0, 1000, 'max_fill')
            model.AddMinEquality(min_fill, list(fill_percent.values()))
            model.AddMaxEquality(max_fill, list(fill_percent.values()))
            
            imbalance = model.NewIntVar(0, 1000, 'fill_imbalance')
            model.Add(imbalance == max_fill - min_fill)

            # Add all equity constraints
            lat_diff = add_equity_constraint(
                model, x, students_df, classes_df,
                student_filter=lambda s: students_df.at[s, 'lat'] == 1,
                class_group=lat_classes,
                label='lat'
            )
            por_diff = add_equity_constraint(
                model, x, students_df, classes_df,
                student_filter=lambda s: students_df.at[s, 'por'] == 1,
                class_group=por_classes,
                label='por'
            )
            n2_diff = add_level_equity_constraint(
                model, x, students_df, classes_df,
                level_val=2,
                label='niv2'
            )
            n3_diff = add_level_equity_constraint(
                model, x, students_df, classes_df,
                level_val=3,
                label='niv3'
            )
            fg_diff = add_gender_balance_constraint(
                model, x, students_df, classes_df,
                class_group=classes_df.index
            )
            b3_diff = add_behavior_equity_constraint(
                model, x, students_df, classes_df,
                level_val=3,
                label='niv3'
            )
            b2_diff = add_behavior_equity_constraint(
                model, x, students_df, classes_df,
                level_val=2,
                label='niv2'
            )

            # Minimize all differences together
            model.Minimize(lat_diff + por_diff + imbalance + n2_diff + n3_diff + fg_diff + b3_diff + b2_diff)

            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                break
                
            if not applied[name]:
                break
                
            rem = random.choice(applied[name])
            applied[name].remove(rem)
            unsatisfied[name].append(rem)

    # Final model with all constraints
    model, x = build_full_model(allowed, classes_df)  # Pass the parameters here too
    
    # Re-add all optimization objectives and constraints
    # ... (same as above) ...

    # Apply all remaining constraints
    for name, _, fn in relax_order:
        for s, other in applied[name]:
            fn(model, x, allowed, students_df, name)

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Échec même après relaxation complète")

    # Extract results
    assign = {s: c for (s,c),var in x.items() if solver.Value(var)}
    broken = {'avec1': [], 'avec2': [], 'sans1': [], 'sans2': []}
    
    for s, row in students_df.iterrows():
        for col, kind in [('avec1','avec1'), ('avec2','avec2')]:
            other = row.get(col)
            if pd.notna(other) and other in assign:
                if assign[s] != assign[other]:
                    broken[kind].append((s, other))
        for col, kind in [('sans1','sans1'), ('sans2','sans2')]:
            other = row.get(col)
            if pd.notna(other) and other in assign:
                if assign[s] == assign[other]:
                    broken[kind].append((s, other))

    return assign, broken