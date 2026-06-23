"""Tests du moteur d'affectation. Lancer avec :  python -m pytest -q"""

import pandas as pd
import pytest

from assign_classes_module import (
    load_data, compute_capacities, build_allowed, describe_no_class, solve,
)


def _prep(students, classes):
    """Pipeline complet : DataFrames bruts -> (allowed, classes_df, students_df)."""
    df_s = pd.DataFrame(students)
    df_c = pd.DataFrame(classes)
    students_df, classes_df, override = load_data(df_s, df_c)
    classes_df = compute_capacities(students_df, classes_df, override)
    allowed = build_allowed(students_df, classes_df)
    return allowed, classes_df, students_df


def _student(name, genre="F", level=1, comp=1, **kw):
    base = {
        "Elèves à affecter": name, "Genre": genre, "por": 0, "lat": 0, "pp": 0,
        "Niveau": level, "Comportement": comp,
        "avec1": None, "avec2": None, "sans1": None, "sans2": None,
    }
    base.update(kw)
    return base


def test_everyone_assigned_within_capacity():
    students = [_student(f"e{i}", genre="F" if i % 2 else "G", level=(i % 3) + 1)
                for i in range(20)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B", "C", "D")]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, broken = solve(allowed, classes_df, students_df, time_limit=10)

    assert len(assignment) == 20
    assert set(assignment.values()) <= {"A", "B", "C", "D"}
    # capacité respectée
    counts = pd.Series(list(assignment.values())).value_counts()
    for c, cap in classes_df["capacity"].items():
        assert counts.get(c, 0) <= cap


def test_friends_kept_together():
    students = [_student("a", avec1="b"), _student("b")]
    students += [_student(f"f{i}") for i in range(8)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B")]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, broken = solve(allowed, classes_df, students_df, time_limit=10)

    assert assignment["a"] == assignment["b"]
    assert broken["avec1"] == []


def test_enemies_kept_apart():
    students = [_student("a", sans1="b"), _student("b")]
    students += [_student(f"f{i}") for i in range(8)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B")]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, broken = solve(allowed, classes_df, students_df, time_limit=10)

    assert assignment["a"] != assignment["b"]
    assert broken["sans1"] == []


def test_deterministic():
    students = [_student(f"e{i}", level=(i % 3) + 1, comp=(i % 3) + 1) for i in range(30)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B", "C")]
    allowed, classes_df, students_df = _prep(students, classes)

    a1, _ = solve(allowed, classes_df, students_df, time_limit=10)
    a2, _ = solve(allowed, classes_df, students_df, time_limit=10)
    assert a1 == a2


def test_impossible_constraint_reported_not_crash():
    # a veut être avec b ET sans b : impossible -> l'un des deux vœux est cassé.
    students = [_student("a", avec1="b", sans1="b"), _student("b")]
    students += [_student(f"f{i}") for i in range(6)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B")]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, broken = solve(allowed, classes_df, students_df, time_limit=10)

    total_broken = sum(len(v) for v in broken.values())
    assert total_broken >= 1


def test_options_respected():
    # élèves POR doivent aller en classe POR (E), élèves LAT en classe LAT (C/D).
    students = [_student("p1", por=1), _student("p2", por=1),
                _student("l1", lat=1), _student("l2", lat=1)]
    students += [_student(f"n{i}") for i in range(6)]
    classes = [
        {"Nom": "A", "por": None, "lat": None, "pp": None, "capacité": None},
        {"Nom": "B", "por": None, "lat": None, "pp": None, "capacité": None},
        {"Nom": "C", "por": None, "lat": 1, "pp": None, "capacité": None},
        {"Nom": "E", "por": 1, "lat": None, "pp": None, "capacité": None},
    ]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, _ = solve(allowed, classes_df, students_df, time_limit=10)

    assert assignment["p1"] == "E" and assignment["p2"] == "E"
    assert assignment["l1"] == "C" and assignment["l2"] == "C"


def test_tech_cat_options_respected():
    # tech -> classe tech (E) ; cat i -> classe cat (C). Contraintes dures.
    students = [_student("t1", tech=1), _student("t2", tech=1),
                _student("c1", **{"cat i": 1}), _student("c2", **{"cat i": 1})]
    students += [_student(f"n{i}") for i in range(6)]
    classes = [
        {"Nom": "A", "por": None, "lat": None, "pp": None, "tech": None, "cat i": None, "capacité": None},
        {"Nom": "B", "por": None, "lat": None, "pp": None, "tech": None, "cat i": None, "capacité": None},
        {"Nom": "C", "por": None, "lat": 1, "pp": None, "tech": None, "cat i": 1, "capacité": None},
        {"Nom": "E", "por": 1, "lat": None, "pp": None, "tech": 1, "cat i": None, "capacité": None},
    ]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, _ = solve(allowed, classes_df, students_df, time_limit=10)

    assert assignment["t1"] == "E" and assignment["t2"] == "E"
    assert assignment["c1"] == "C" and assignment["c2"] == "C"


def test_incompatible_options_reported_no_class():
    # Un élève veut lat ET tech mais aucune classe ne combine les deux.
    students = [_student("x", lat=1, tech=1)]
    students += [_student(f"n{i}") for i in range(6)]
    classes = [
        {"Nom": "A", "por": None, "lat": None, "pp": None, "tech": None, "cat i": None, "capacité": None},
        {"Nom": "C", "por": None, "lat": 1, "pp": None, "tech": None, "cat i": None, "capacité": None},
        {"Nom": "E", "por": None, "lat": None, "pp": None, "tech": 1, "cat i": None, "capacité": None},
    ]
    allowed, classes_df, students_df = _prep(students, classes)

    assert allowed["x"] == []  # aucune classe possible
    problems = describe_no_class(students_df, classes_df)
    assert "x" in problems
    assert set(problems["x"]["options"]) == {"lat", "tech"}


def test_pap_balanced_across_classes():
    # 4 élèves PAP, 2 classes -> répartition équilibrée (2/2).
    students = [_student(f"p{i}", pap=1) for i in range(4)]
    students += [_student(f"n{i}") for i in range(4)]
    classes = [{"Nom": c, "por": None, "lat": None, "pp": None, "capacité": None}
               for c in ("A", "B")]
    allowed, classes_df, students_df = _prep(students, classes)

    assignment, _ = solve(allowed, classes_df, students_df, time_limit=10)

    pap_per_class = {}
    for s in students_df.index:
        if students_df.at[s, "pap"] == 1:
            pap_per_class[assignment[s]] = pap_per_class.get(assignment[s], 0) + 1
    assert max(pap_per_class.values()) - min(pap_per_class.values()) <= 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
