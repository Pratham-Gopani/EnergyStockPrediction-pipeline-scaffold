"""Tests for nlp.headline_strength: anchor scores must hit the spec's example
headlines, and inflected verb forms ("resigns", not just "resign") must match.
"""

from __future__ import annotations

from nlp.headline_strength import FLOOR_SCORE, headline_strength


def test_ceo_resignation_scores_098():
    assert headline_strength("Energy Corp CEO resigns amid controversy") == 0.98
    assert headline_strength("CEO resigns") == 0.98


def test_inflected_verb_forms_match_not_just_bare_infinitive():
    # A `resign\b` pattern would NOT match "resigns"/"resigned" -- this is the
    # exact gotcha called out in SKILL.md.
    assert headline_strength("Chairman resigned after board meeting") == 0.98
    assert headline_strength("MD quits after allegations") == 0.98
    assert headline_strength("CFO exits the company") == 0.98


def test_major_order_win_scores_096():
    assert headline_strength("Company Wins Major Order Worth Rs 5000 Crore") == 0.96
    assert headline_strength("Firm bags mega deal with government") == 0.96


def test_quarterly_earnings_scores_093():
    assert headline_strength("Company Reports Q2 Results, Net Profit Jumps 20%") == 0.93


def test_dividend_scores_075():
    assert headline_strength("Company announces final dividend of Rs 5 per share") == 0.75


def test_general_commentary_hits_floor():
    assert headline_strength("Analysts discuss sector outlook for next year") == FLOOR_SCORE


def test_empty_headline_hits_floor():
    assert headline_strength("") == FLOOR_SCORE
    assert headline_strength(None) == FLOOR_SCORE
