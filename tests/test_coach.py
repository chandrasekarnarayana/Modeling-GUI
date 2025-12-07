from modeling_gui.coach import CoachManager, CoachState


def test_coach_transitions():
    coach = CoachManager()
    assert "Step 1" in coach.update(CoachState.DATA_UNLOADED)
    assert "choose" in coach.update(CoachState.DATA_LOADED).lower()
    assert "ready" in coach.update(CoachState.TARGET_SELECTED).lower()
    assert "analyzing" in coach.update(CoachState.ANALYSIS_RUNNING).lower()
    assert "trained" in coach.update(CoachState.ANALYSIS_DONE).lower()
    assert "wrong" in coach.update(CoachState.ERROR).lower()
