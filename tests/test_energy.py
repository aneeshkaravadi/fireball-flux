"""The optical-to-impact energy relation against the published coefficients and the CNEOS column.

These checks show the code reproduces how CNEOS builds its impact-energy column. They are not an
independent test of whether those impact energies are right.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from fireballs import data, energy

FIX = Path(__file__).parent / "fixtures" / "fireball_api_sample.json"


def test_published_coefficients():
    # CNEOS intro page: E = 8.2508 * E0^0.885 (kt), 1 kt = 4.185e12 J
    assert (energy.BROWN_C, energy.BROWN_D, energy.KT_J) == (8.2508, 0.885, 4.185e12)
    # the same fit written as a luminous efficiency, tau = 0.1212 * E0^0.115 (Hibberd 2023 quoting Brown 2002)
    assert 1.0 / energy.TAU_A == pytest.approx(energy.BROWN_C, rel=1e-4)
    assert 1.0 - energy.TAU_B == pytest.approx(energy.BROWN_D, abs=1e-12)
    e0 = np.logspace(-3, 3, 13)
    assert energy.impact_energy_kt(e0 * energy.KT_J) * energy.luminous_efficiency(e0 * energy.KT_J) == pytest.approx(e0, rel=1e-4)


def test_chelyabinsk_numbers_on_the_cneos_page():
    # "3.75 x 10^14 Joules ... about 90 kilotons ... total impact energy ... 440 kt" (CNEOS news, 2013-03-01)
    assert 3.75e14 / energy.KT_J == pytest.approx(90, rel=0.01)
    assert energy.impact_energy_kt(3.75e14) == pytest.approx(440, rel=0.005)


def test_inverse():
    e = np.array([0.05, 0.3, 1.0, 17.0, 440.0])
    assert energy.impact_energy_kt(energy.radiated_energy_J(e)) == pytest.approx(e, rel=1e-12)


def test_round_sig():
    assert energy.round_sig([0.07349, 0.07351, 441.2, 12.6]) == pytest.approx([0.073, 0.074, 440.0, 13.0])


def test_fixture_rows_match_the_column():
    df = data.tidy(data.parse_api(json.loads(FIX.read_text())))
    rel, match = energy.check_column(df.radiated_J, df.impact_kt)
    assert match.all()
    assert np.abs(rel).max() < 0.05  # two significant figures
