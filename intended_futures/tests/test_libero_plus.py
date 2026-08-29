from pathlib import Path

import numpy as np

from intended_futures.libero_plus import (
    parse_initial_supports,
    parse_language,
    parse_layout_name,
    parse_region_centers,
    resolve_subject_xy,
)


FIXTURE = """
(define (problem LIBERO_Tabletop_Manipulation)
  (:domain robosuite)
  (:language Pick the bowl next to the ramekin and place it on the plate)
    (:regions
      (box_region
          (:target main_table)
          (:ranges (
              (0.06 0.02 0.08 0.04)
            )
          )
      )
      (stove_region
          (:target main_table)
          (:ranges (
              (-0.42 -0.15 -0.40 -0.13)
            )
          )
      )
      (cook_region
          (:target flat_stove_1)
      )
    )
  (:fixtures
    main_table - table
    flat_stove_1 - flat_stove
  )
  (:objects
    akita_black_bowl_1 - akita_black_bowl
    cookies_1 - cookies
  )
  (:init
    (On akita_black_bowl_1 flat_stove_1_cook_region)
    (On cookies_1 main_table_box_region)
    (On flat_stove_1 main_table_stove_region)
  )
  (:goal
    (And (On akita_black_bowl_1 plate_1))
  )
)
"""


def test_layout_name_and_bddl_geometry_parsing():
    family, level, sample = parse_layout_name("family_level5_sample4")
    assert (family, level, sample) == ("family", 5, 4)
    assert parse_language(FIXTURE).startswith("Pick the bowl")
    centers = parse_region_centers(FIXTURE)
    np.testing.assert_allclose(centers["box_region"], [0.07, 0.03])
    supports = parse_initial_supports(FIXTURE)
    np.testing.assert_allclose(
        resolve_subject_xy("akita_black_bowl_1", supports, centers),
        [-0.41, -0.14],
    )


def test_parser_accepts_realistic_scientific_notation(tmp_path: Path):
    text = FIXTURE.replace(
        "(0.06 0.02 0.08 0.04)",
        "(-8.0e-3 3e-1 2.0e-2 0.32)",
    )
    centers = parse_region_centers(text)
    np.testing.assert_allclose(centers["box_region"], [0.006, 0.31])
