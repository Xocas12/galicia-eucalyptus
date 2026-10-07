"""`mfe_class` against the vocabulary the real MFE shapefiles actually use.

The formation strings below are copied from the two editions as they ship, not invented. MFE25
for Galicia (MFE_11.shp, 96,665 polygons) writes three of its four pine formations in the
singular and its riparian class in the singular, where MFE50 pluralises both. The rules used to
match the plural exactly, so 174,475 ha of pine and 21,391 ha of riparian forest were classed
255 and dropped from the reference the species map is scored against.

The areas in the comments are the hectares each formation covers in MFE25 Galicia, measured
from the shapefile, and are what makes a silent miss expensive.
"""

from __future__ import annotations

import pytest

from eucalyptus_impact.real.reference import (
    MFE_FORMATION_FIELDS,
    MFE_USE_FIELDS,
    mfe_class,
)

EUC, PINE, NATIVE, SHRUB, AGRI, OTHER, EXCLUDED = 0, 1, 2, 3, 4, 5, 255

# (formation, use, expected class, hectares in MFE25 Galicia)
MFE25_REAL = [
    ("Eucaliptales", "Arbolado", EUC, 284_241),
    # the three that used to be missed: singular "Pinar", not "Pinares"
    ("Pinar de pino radiata", "Arbolado", PINE, 87_705),
    ("Pinar de pino pinaster en región mediterránea", "Arbolado", PINE, 48_249),
    ("Pinar de pino albar (Pinus sylvestris)", "Arbolado", PINE, 38_521),
    ("Pinares de pino pinaster en región atlántica", "Arbolado", PINE, 217_461),
    ("Robledales de Q. robur y/o Q. petraea", "Arbolado", NATIVE, 128_741),
    ("Bosques mixtos de frondosas autóctonas en región atlántica", "Arbolado", NATIVE, 116_978),
    ("Melojares (Quercus pyrenaica)", "Arbolado", NATIVE, 78_888),
    ("Castañares (Castanea sativa)", "Arbolado", NATIVE, 26_889),
    # singular "Bosque ribereño", where MFE50 has "Bosques ribereños"
    ("Bosque ribereño", "Arbolado", NATIVE, 21_391),
    ("Abedulares (Betula spp.)", "Arbolado", NATIVE, 12_253),
    ("Encinares (Quercus ilex)", "Arbolado", NATIVE, 2_767),
    ("No arbolado", "Desarbolado", SHRUB, 586_455),
    ("No arbolado", "Cultivos", AGRI, 825_176),
    ("No arbolado", "Artificial", OTHER, 81_781),
    ("No arbolado", "Agua", OTHER, 20_176),
]

# MFE50 spellings, which must keep working
MFE50_REAL = [
    ("Eucaliptales", "Arbolado", EUC),
    ("Pinares de pino pinaster en región atlántica", "Arbolado", PINE),
    ("Robledal", "Arbolado", NATIVE),
    ("Melojar", "Arbolado", NATIVE),
    ("Castañar", "Arbolado", NATIVE),
    ("Bosques ribereños", "Arbolado", NATIVE),
    ("Abedular", "Arbolado", NATIVE),
    ("Encinar", "Arbolado", NATIVE),
]

# Mixed, sparse and non-native formations have no single pixel class and stay excluded.
MUST_STAY_EXCLUDED = [
    ("Otras especies de producción en mezcla", "Arbolado", 168_815),
    ("Mezcla de coníferas y frondosas autóctonas en región atlántica", "Arbolado", 75_614),
    ("Frondosas alóctonas con  autóctonas", "Arbolado", 29_138),
    ("Mezcla de coníferas autoctonas con alóctonas", "Arbolado", 3_210),
    ("Frondosas alóctonas invasoras", "Arbolado", 2_582),
    ("No arbolado", "Arbolado ralo", 56_232),
]


@pytest.mark.parametrize(("formation", "use", "expected", "hectares"), MFE25_REAL)
def test_mfe25_vocabulary(formation, use, expected, hectares):
    assert mfe_class(formation, use) == expected, (
        f"{formation!r} -> {mfe_class(formation, use)}, expected {expected}; "
        f"{hectares:,} ha of Galicia ride on this row"
    )


@pytest.mark.parametrize(("formation", "use", "expected"), MFE50_REAL)
def test_mfe50_vocabulary_still_works(formation, use, expected):
    assert mfe_class(formation, use) == expected


@pytest.mark.parametrize(("formation", "use", "hectares"), MUST_STAY_EXCLUDED)
def test_mixed_and_sparse_stay_excluded(formation, use, hectares):
    """Over-matching is as bad as under-matching: a mixed polygon has no single pixel class."""
    assert mfe_class(formation, use) == EXCLUDED, (
        f"{formation!r} became a class; it is mixed or sparse and must stay excluded "
        f"({hectares:,} ha)"
    )


def test_the_pine_rule_catches_both_numbers():
    """The regression this file exists for."""
    assert mfe_class("Pinar de pino radiata", "Arbolado") == PINE
    assert mfe_class("Pinares de pino pinaster en región atlántica", "Arbolado") == PINE


def test_the_riparian_rule_catches_both_numbers():
    assert mfe_class("Bosque ribereño", "Arbolado") == NATIVE
    assert mfe_class("Bosques ribereños", "Arbolado") == NATIVE


def test_accents_and_case_do_not_decide_the_class():
    """An edition that drops a tilde or shouts must not lose a forest type."""
    assert mfe_class("PINAR DE PINO RADIATA", "ARBOLADO") == PINE
    assert mfe_class("Bosque riberenio".replace("renio", "reno"), "Arbolado") == NATIVE
    assert mfe_class("Castanares (Castanea sativa)", "Arbolado") == NATIVE


def test_unknown_and_empty_are_excluded_rather_than_guessed():
    assert mfe_class(None, None) == EXCLUDED
    assert mfe_class("", "") == EXCLUDED
    assert mfe_class("Something nobody has seen", "Arbolado") == EXCLUDED


def test_both_editions_column_names_are_declared():
    """The reader picks the column by edition; losing one breaks that edition silently."""
    assert "NOM_FORARB" in MFE_FORMATION_FIELDS and "FormArbol" in MFE_FORMATION_FIELDS
    assert "USOS_GENER" in MFE_USE_FIELDS and "UsoMFE" in MFE_USE_FIELDS
