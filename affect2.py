# Script complet corrigé : assign_classes_with_override.py

import pandas as pd
from ortools.sat.python import cp_model
import random
import os
import matplotlib.pyplot as plt
from xlsxwriter.utility import xl_col_to_name

# 1. Chargement des CSV
students_df = pd.read_excel('Liste.xlsx', sheet_name='liste')
classes_df = pd.read_excel('Liste.xlsx', sheet_name='classes')

# 2. Renommages
students_df = students_df.rename(columns={
    'Elèves à affecter': 'student',
    'Niveau': 'level'
})
classes_df = classes_df.rename(columns={
    'Nom': 'class_id',
    'capacité': 'capacity_override'
})

# 3. Indexation
students_df['student'] = students_df['student'].astype(str)
students_df.set_index('student', inplace=True)
classes_df['class_id'] = classes_df['class_id'].astype(str)
classes_df.set_index('class_id', inplace=True)

# 4. Flags PP/POR/LAT et overrides
for col in ['pp', 'por', 'lat']:
    # if missing, create it; then fillna+astype
    if col not in students_df.columns:
        students_df[col] = 0
    students_df[col] = students_df[col].fillna(0).astype(int)

    if col not in classes_df.columns:
        classes_df[col] = 0
    classes_df[col] = classes_df[col].fillna(0).astype(int)

# Conserver override uniquement pour les classes qui en ont
classes_df['capacity_override'] = pd.to_numeric(
    classes_df.get('capacity_override', pd.Series()), errors='coerce'
)
override_map = {
    cid: int(val)
    for cid, val in classes_df['capacity_override'].items()
    if not pd.isna(val)
}

# 5. Comptages
total_students = len(students_df)
count_pp  = int(students_df['pp'].sum())

# 6. Calcul des capacités finales
final_caps = {}

# 6.1 Répartition PP strictement entre classes PP sans override
pp_classes = [c for c, v in classes_df['pp'].items() if v == 1]
pp_no_override = [c for c in pp_classes if c not in override_map]
if pp_no_override:
    base_pp = count_pp // len(pp_no_override)
    extra_pp = count_pp % len(pp_no_override)
    for idx, c in enumerate(sorted(pp_no_override)):
        final_caps[c] = base_pp + (1 if idx < extra_pp else 0)
# On ajoute les overrides pour PP
for c, cap in override_map.items():
    final_caps[c] = cap

# 6.2 Répartition uniforme du reste
assigned_pp = sum(final_caps[c] for c in pp_classes if c in final_caps)
remaining = total_students - assigned_pp
# Classes restantes (non-PP et non-override)
uniform_classes = [
    c for c in classes_df.index
    if c not in pp_classes and c not in override_map
]
if uniform_classes:
    base = remaining // len(uniform_classes)
    extra = remaining % len(uniform_classes)
    for idx, c in enumerate(sorted(uniform_classes)):
        final_caps[c] = base + (1 if idx < extra else 0)
else:
    if remaining != 0:
        raise ValueError(f"Impossible de répartir {remaining} élèves restants")

classes_df['capacity'] = classes_df.index.map(final_caps)
print("Capacités finales :")
print(classes_df['capacity'])
# 6.3 Classes avec override
# 7. Définition des classes autorisées par élève
groups = {
    'pp': [c for c, v in classes_df['pp'].items() if v == 1],
    'por': [c for c, v in classes_df['por'].items() if v == 1],
    'lat': [c for c, v in classes_df['lat'].items() if v == 1],
}
non_pp = [c for c in classes_df.index if c not in groups['pp']]

# initial per‐type lists
pp_classes  = [c for c, v in classes_df['pp'].items()  if v == 1]
por_classes = [c for c, v in classes_df['por'].items() if v == 1]
lat_classes = [c for c, v in classes_df['lat'].items() if v == 1]

# fallback: if any list is empty, let those students choose *any* class
all_classes = classes_df.index.tolist()
if not pp_classes:
    print("⚠️ Aucune classe PP trouvée → tous les PP peuvent aller dans n'importe quelle classe")

if not por_classes:
    print("⚠️ Aucune classe POR trouvée → tous les POR peuvent aller dans n'importe quelle classe")

if not lat_classes:
    print("⚠️ Aucune classe LAT trouvée → tous les LAT peuvent aller dans n'importe quelle classe")

# now recompute the “dual” and “non‐PP” groups
dual_pp_por_classes = [c for c in pp_classes if c in por_classes]
non_pp_classes      = [c for c in classes_df.index if c not in pp_classes]

# and finally your allowed‐dict loop stays exactly the same:
allowed = {}
for s, row in students_df.iterrows():
    is_pp  = row['pp']  == 1
    is_por = row['por'] == 1
    is_lat = row['lat'] == 1

    if is_pp and is_por:
        allowed[s] = dual_pp_por_classes
    elif is_pp:
        allowed[s] = pp_classes
    elif is_por:
        # exclude dual from POR‐only
        allowed[s] = [c for c in por_classes if c not in dual_pp_por_classes]
    elif is_lat:
        allowed[s] = lat_classes
    else:
        allowed[s] = non_pp_classes


# Vérification 1 : capacité totale vs nombre d'élèves
total_students = len(students_df)
total_capacity = sum(final_caps.values())
print(f"Total students = {total_students}, total capacity = {total_capacity}")
if total_capacity < total_students:
    raise RuntimeError(f"Capacité totale insuffisante : {total_capacity} < {total_students}")

# Vérification 2 : aucun élève sans classes autorisées
for s, cls_list in allowed.items():
    if len(cls_list) == 0:
        print("⚠️ Élève sans affectation possible :", s)

equity_classes = [
    c for c in classes_df.index
    if classes_df.at[c, 'pp'] == 0  # Include all non-PP
]




def build_base_model(x, allowed, classes_df, students_df):
    """Construit le modèle vide avec variables, 1 classe par élève et capacités."""
    model = cp_model.CpModel()
    # 1. Variables et « 1 élève → 1 classe »
    for s, cls_list in allowed.items():
        for c in cls_list:
            x[s, c] = model.NewBoolVar(f"x_{s}_{c}")
        model.Add(sum(x[s, c] for c in cls_list) == 1)
    # 2. Contraintes de capacité
    for c, cap in classes_df['capacity'].items():
        model.Add(sum(var for (s2,c2), var in x.items() if c2 == c) <= cap)
    return model

def add_enemy_constraints(model, x, allowed, students_df, field):
    """Ajoute les contraintes « sans1 » ou « sans2 » (hard)."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] + x[other, c] <= 1)

def add_friend_constraints(model, x, allowed, students_df, field):
    """Ajoute les contraintes « avec1 » ou « avec2 » (potentiellement hard)."""
    for s, row in students_df.iterrows():
        other = row.get(field)
        if pd.notna(other) and other in students_df.index:
            common = set(allowed[s]) & set(allowed[other])
            for c in common:
                model.Add(x[s, c] == x[other, c])

# 1️⃣ Préparez vos listes de contraintes à appliquer
enemy1 = [(s, row['sans1']) for s, row in students_df.iterrows() if pd.notna(row['sans1'])]
enemy2 = [(s, row['sans2']) for s, row in students_df.iterrows() if pd.notna(row['sans2'])]
friend1 = [(s, row['avec1']) for s, row in students_df.iterrows() if pd.notna(row['avec1'])]
friend2 = [(s, row['avec2']) for s, row in students_df.iterrows() if pd.notna(row['avec2'])]

# L’ordre de relaxation (du plus laxiste au plus stricte)
relax_order = [
    ('avec2', friend2, add_friend_constraints),
    ('avec1', friend1, add_friend_constraints),
    ('sans2', enemy2,   add_enemy_constraints),
    ('sans1', enemy1,   add_enemy_constraints),
]

# 2️⃣ Fonction de résolution avec relaxation hiérarchique
import random
from ortools.sat.python import cp_model

def solve_hierarchical(allowed, classes_df, students_df):
    # On stocke les paires initiales pour chaque niveau
    enemy1 = [(s, row['sans1']) for s,row in students_df.iterrows() if pd.notna(row['sans1'])]
    enemy2 = [(s, row['sans2']) for s,row in students_df.iterrows() if pd.notna(row['sans2'])]
    friend1 = [(s, row['avec1']) for s,row in students_df.iterrows() if pd.notna(row['avec1'])]
    friend2 = [(s, row['avec2']) for s,row in students_df.iterrows() if pd.notna(row['avec2'])]
    relax_order = [
        ('avec2', friend2, add_friend_constraints),
        ('avec1', friend1, add_friend_constraints),
        ('sans2', enemy2,   add_enemy_constraints),
        ('sans1', enemy1,   add_enemy_constraints),
    ]

    # On va manipuler une copie modifiable de ces listes
    applied = {name: pairs.copy() for name, pairs, _ in relax_order}
    unsatisfied = {name: [] for name,_,_ in relax_order}
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5

    # Fonction interne pour rebuild le modèle complet
    def build_full_model():
        x = {}
        m = cp_model.CpModel()
        # 1) variables et 1 élève→1 classe
        for s, cls_list in allowed.items():
            for c in cls_list:
                x[s,c] = m.NewBoolVar(f"x_{s}_{c}")
            m.Add(sum(x[s,c] for c in cls_list) == 1)
        # 2) capacités
        for c,cap in classes_df['capacity'].items():
            m.Add(sum(var for (s2,c2),var in x.items() if c2==c) <= cap)
        # 3) ennemis (hard, toujours)
        add_enemy_constraints(m, x, allowed, students_df, 'sans1')
        add_enemy_constraints(m, x, allowed, students_df, 'sans2')
        return m, x

    def add_equity_constraint(model, x, students_df, classes_df, student_filter, class_group, label):
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
    def add_gender_balance_constraint(model, x, students_df, classes_df, class_group):
        diffs = []
        for c in class_group:
            cap = classes_df.at[c, 'capacity']

            f_count = model.NewIntVar(0, cap, f'f_{c}')
            g_count = model.NewIntVar(0, cap, f'g_{c}')
            total_count = model.NewIntVar(0, cap, f'total_{c}')

            model.Add(f_count == sum(x[s, c] for s in students_df.index if (s, c) in x and students_df.at[s, 'Genre'] == 'F'))
            model.Add(g_count == sum(x[s, c] for s in students_df.index if (s, c) in x and students_df.at[s, 'Genre'] == 'G'))
            model.Add(total_count == f_count + g_count)

            f_pct = model.NewIntVar(0, 1000, f'f_pct_{c}')
            g_pct = model.NewIntVar(0, 1000, f'g_pct_{c}')

            # Division par le nombre total d'élèves dans la classe
            model.AddDivisionEquality(f_pct, f_count * 1000, cap)
            model.AddDivisionEquality(g_pct, g_count * 1000, cap)

            diff = model.NewIntVar(0, 1000, f'diff_{c}')
            model.AddAbsEquality(diff, f_pct - g_pct)
            diffs.append(diff)

        max_diff = model.NewIntVar(0, 1000, 'fg_max_diff')
        model.AddMaxEquality(max_diff, diffs)
        return max_diff

    def add_behavior_equity_constraint(model, x, students_df, classes_df, level_val, label):
        fill = {}
        for c in classes_df.index:
            cap = classes_df.at[c, 'capacity']
            count = model.NewIntVar(0, cap, f'{label}_count_{c}')
            model.Add(
                count == sum(
                    x[s, c]
                    for s in students_df.index
                    if (s, c) in x and students_df.at[s, 'Comportement'] == level_val
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
        fill = {}
        for c in classes_df.index:
            cap = classes_df.at[c, 'capacity']
            count = model.NewIntVar(0, cap, f'{label}_count_{c}')
            model.Add(
                count == sum(
                    x[s, c]
                    for s in students_df.index
                    if (s, c) in x and students_df.at[s, 'level'] == level_val
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

    # itération sur chaque niveau
    for name, _, adder in relax_order:
        # on essaie d’appliquer toutes les paires encore « applied[name] »
        while True:
            model, x = build_full_model()
            # on applique TOUTES les paires déjà validées des niveaux précédents
            for lvl, pairs, fn in relax_order:
                if lvl == name:
                    break
                for s, other in applied[lvl]:
                    fn(model, x, allowed, students_df, lvl)
            # puis on applique tout ce niveau
            for s, other in applied[name]:
                adder(model, x, allowed, students_df, name)
            # solve
            status = solver.Solve(model)
            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                # ce niveau passe, on peut sortir du while et passer au suivant
                break
            # sinon : on retire une paire au hasard de applied[name]
            if not applied[name]:
                # plus aucune paire à retirer : on abandonne
                break
            rem = random.choice(applied[name])
            applied[name].remove(rem)
            unsatisfied[name].append(rem)
            # et on retente (le while boucle)

    # au terme, on reconstruit une dernière fois le modèle complet
    model, x = build_full_model()
    
    fill_percent = {}
    for c in equity_classes:
        cap = classes_df.at[c, 'capacity']
        count = model.NewIntVar(0, cap, f'count_{c}')
        model.Add(count == sum(x[s, c] for s in students_df.index if (s, c) in x))
        scaled = model.NewIntVar(0, 1000, f'fill_{c}')
        model.AddDivisionEquality(scaled, count * 1000, cap)
        fill_percent[c] = scaled

    min_fill = model.NewIntVar(0, 1000, 'min_fill')
    max_fill = model.NewIntVar(0, 1000, 'max_fill')
    model.AddMinEquality(min_fill, list(fill_percent.values()))
    model.AddMaxEquality(max_fill, list(fill_percent.values()))
    model.Add(max_fill >= min_fill)

    imbalance = model.NewIntVar(0, 1000, 'fill_imbalance')
    model.Add(imbalance == max_fill - min_fill)
    
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
    model.Minimize(lat_diff + por_diff + imbalance + n2_diff + n3_diff + fg_diff + b3_diff + b2_diff)
    # on ré-applique toutes les paires de tous les niveaux sauf celles « unsatisfied »
    for name, _, fn in relax_order:
        for s, other in applied[name]:
            fn(model, x, allowed, students_df, name)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("Échec même après relaxation complète")
    # extraction finale
    assign = {s: c for (s,c),var in x.items() if solver.Value(var)}
    broken = {'avec1': [], 'avec2': [], 'sans1': [], 'sans2': []}
    for s, row in students_df.iterrows():
        for col, kind in [('avec1','avec1'), ('avec2','avec2')]:
            other = row.get(col)
            if pd.notna(other) and other in assign:
                # friend constraint violated if they end up in different classes
                if assign[s] != assign[other]:
                    broken[kind].append((s, other))
        for col, kind in [('sans1','sans1'), ('sans2','sans2')]:
            other = row.get(col)
            if pd.notna(other) and other in assign:
                # enemy constraint violated if they end up in the same class
                if assign[s] == assign[other]:
                    broken[kind].append((s, other))

    return assign, broken


# 3️⃣ Appel de la fonction
assignment, broken = solve_hierarchical(allowed, classes_df, students_df)
# Calculate satisfied constraints
all_constraints = {
    'avec1': [(s, row['avec1']) for s, row in students_df.iterrows() if pd.notna(row['avec1'])],
    'avec2': [(s, row['avec2']) for s, row in students_df.iterrows() if pd.notna(row['avec2'])],
    'sans1': [(s, row['sans1']) for s, row in students_df.iterrows() if pd.notna(row['sans1'])],
    'sans2': [(s, row['sans2']) for s, row in students_df.iterrows() if pd.notna(row['sans2'])],
}

satisfied = {}
# To avoid duplicates like (A,B) and (B,A), we use sets
for key, pairs in all_constraints.items():
    broken_set = set(tuple(sorted(p)) for p in broken.get(key, []))
    satisfied_set = set(tuple(sorted(p)) for p in pairs) - broken_set
    satisfied[key] = list(satisfied_set)
# To avoid duplicates like (A,B) and (B,A), we use sets
for key, pairs in all_constraints.items():
    broken_set = set(tuple(sorted(p)) for p in broken.get(key, []))
    satisfied_set = set(tuple(sorted(p)) for p in pairs) - broken_set
    satisfied[key] = list(satisfied_set)

# 2. Préparer les DataFrames
students = students_df.copy()
students['classe'] = [assignment[s] for s in students.index]
students.reset_index(inplace=True)
students_sorted = students.sort_values(['classe','student'])
# Feuille "Classes"
classes_sheet = students_sorted.copy()

# Feuille "Impossibilites"
# On ne garde que les listes non vides
data = {}
for lvl, pairs in broken.items():
    if pairs:
        # Separate source and other into two columns
        s_list, o_list = zip(*pairs)
        data[f"{lvl}_s"] = list(s_list) + [''] * (max(len(pairs) for pairs in broken.values()) - len(s_list))
        data[f"{lvl}_o"] = list(o_list) + [''] * (max(len(pairs) for pairs in broken.values()) - len(o_list))

# Create the dataframe
imposs_df = pd.DataFrame(data)

# Format satisfied constraints like Impossibilites
satisfied_data = {}
max_len = max((len(pairs) for pairs in satisfied.values() if pairs), default=0)

for constraint, pairs in satisfied.items():
    if pairs:
        s_list, o_list = zip(*pairs)
        satisfied_data[f"{constraint}_s"] = list(s_list) + [''] * (max_len - len(s_list))
        satisfied_data[f"{constraint}_o"] = list(o_list) + [''] * (max_len - len(o_list))

contraintes_df = pd.DataFrame(satisfied_data)

# Feuille "Dashboards": résumé par classe

summary = students_sorted.groupby('classe').agg(
    Total=('student','count'),
    Niveau1=('level', lambda x: (x==1).sum()),
    Niveau2=('level', lambda x: (x==2).sum()),
    Niveau3=('level', lambda x: (x==3).sum()),
    POR=('por', 'sum'),
    LAT=('lat', 'sum'),
    Filles=('Genre', lambda x: (x=='F').sum()),
    Garçons=('Genre', lambda x: (x=='G').sum()),
    Comp1=('Comportement', lambda x: (x==1).sum()),
    Comp2=('Comportement', lambda x: (x==2).sum()),
    Comp3=('Comportement', lambda x: (x==3).sum()),
)
# Pourcentages
summary['%N1'] = (summary['Niveau1'] / summary['Total'] * 100).round(0)
summary['%N2'] = (summary['Niveau2'] / summary['Total'] * 100).round(0)
summary['%N3'] = (summary['Niveau3'] / summary['Total'] * 100).round(0)
summary['%Filles'] = (summary['Filles'] / summary['Total'] * 100).round(0)
summary['%Garçons'] = (summary['Garçons'] / summary['Total'] * 100).round(0)
summary['%C1'] = (summary['Comp1'] / summary['Total'] * 100).round(0)
summary['%C2'] = (summary['Comp2'] / summary['Total'] * 100).round(0)
summary['%C3'] = (summary['Comp3'] / summary['Total'] * 100).round(0)


# Sheet "Tableau" — one column per class with students listed under
tableau_data = {
    cls: list(grp['student']) for cls, grp in students_sorted.groupby('classe')
}
max_len = max(len(lst) for lst in tableau_data.values())

# Pad columns to have equal lengths
tableau_padded = {
    cls: lst + [''] * (max_len - len(lst)) for cls, lst in tableau_data.items()
}

tableau_df = pd.DataFrame(tableau_padded)

# 4. Export Excel avec tableaux et images
from xlsxwriter.utility import xl_col_to_name

output_path = 'assignments.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    # 1) Classes
    classes_sheet.to_excel(writer, sheet_name='Classes', index=False)

    # 2) Impossibilités
    imposs_df.to_excel(writer, sheet_name='Impossibilites', index=False)

    # 3) Contraintes
    contraintes_df.to_excel(writer, sheet_name='Contraintes', index=False)

    # 4) Tableau (custom)


   # Write the student list first (starting from row 8 to leave room for headers/sparklines)
    tableau_df.to_excel(writer, sheet_name='Tableau', startrow=7, startcol=1 ,index=False)
    ws = writer.sheets['Tableau']



    classes = tableau_df.columns.tolist()
    n = len(tableau_df)

    print(n)



    # 0-based rows
    HDR       = 0   # "Id", class names
    FG_LABEL  = 1   # "F / G"
    FG_CHART  = 2   # sparkline here
    LVL_LABEL = 3   # "N1 N2 N3"
    LVL_CHART = 4   # sparkline here
    COMP_LABEL = 5   # "c1 c2 c3"
    COMP_CHART = 6   # sparkline here
    PORLAT_LABEL = 7 # "POR / LAT"
    STU_START = 8   # students begin here

    # 1) Headers
    ws.write(HDR, 0, "Id")
    for j, cl in enumerate(classes):
        ws.write(HDR, j+1, cl)

        # 2) Labels
        ws.write(FG_LABEL,  j+1, "       F            G")
        ws.write(LVL_LABEL, j+1, "   N1      N2     N3")
        ws.write(COMP_LABEL, j+1, "   C1      C2     C3")
        por = summary.loc[cl, 'POR']
        lat = summary.loc[cl, 'LAT']
        ws.write(PORLAT_LABEL, j+1, f"POR: {int(por)} / LAT: {int(lat)}")

    # 3) Students numbers
    for i in range(n):
        r = STU_START + i
        ws.write(r, 0, i+1)


    # 4) Raw data deep below
    BASE = 100
    fg1  = BASE
    fg2  = BASE+1
    lvl1 = BASE+2
    lvl2 = BASE+3
    lvl3 = BASE+4
    com1 = BASE+5
    com2 = BASE+6
    com3 = BASE+7

    for j, cl in enumerate(classes):
        ws.write(fg1,  j+1, summary.loc[cl,'Filles'])
        ws.write(fg2,  j+1, summary.loc[cl,'Garçons'])
        ws.write(lvl1, j+1, summary.loc[cl,'Niveau1'])
        ws.write(lvl2, j+1, summary.loc[cl,'Niveau2'])
        ws.write(lvl3, j+1, summary.loc[cl,'Niveau3'])
        ws.write(com1, j+1, summary.loc[cl,'Comp1'])
        ws.write(com2, j+1, summary.loc[cl,'Comp2'])
        ws.write(com3, j+1, summary.loc[cl,'Comp3'])


    # 5) F/G sparkline at row 2
    for j in range(len(classes)):
        col = xl_col_to_name(j+1)
        rng = f"{col}{fg1+1}:{col}{fg2+1}"
        ws.add_sparkline(FG_CHART, j+1, {
            'range': rng, 'type': 'column',
            'min': 0,
            'first_point': True,
            'last_point':  True,
        })

    # 6) N1-N3 sparkline at row 4
    for j in range(len(classes)):
        col = xl_col_to_name(j+1)
        rng = f"{col}{lvl1+1}:{col}{lvl3+1}"
        ws.add_sparkline(LVL_CHART, j+1, {
            'range': rng, 'type': 'column',
            'min': 0,
            'first_point': True,
            'last_point':  True,
        })

    # 6bis) C11-C3 sparkline at row 6
    for j in range(len(classes)):
        col = xl_col_to_name(j+1)
        rng = f"{col}{com1+1}:{col}{com3+1}"
        ws.add_sparkline(COMP_CHART, j+1, {
            'range': rng, 'type': 'column',
            'min': 0,
            'first_point': True,
            'last_point':  True,
        })

    # 7) Tidy
    ws.set_row(FG_LABEL, 15)
    ws.set_row(FG_CHART, 30)
    ws.set_row(LVL_LABEL,15)
    ws.set_row(LVL_CHART,30)
    ws.set_row(COMP_LABEL,15)
    ws.set_row(COMP_CHART,30)
    ws.set_column(0,0,5)
    ws.set_column(1,len(classes),15)

   
    # 5) Dashboards
    summary.to_excel(writer, sheet_name='Dashboards')

print(f"✅ Fichier Excel généré : {output_path}")
