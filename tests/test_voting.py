from crisp.voting import weighted_vote, majority_vote


def test_weighted_vote():
    neighbors = [
        {"label": "A", "similarity": 0.9},
        {"label": "A", "similarity": 0.8},
        {"label": "B", "similarity": 0.95},
    ]
    result = weighted_vote(neighbors)
    assert result["predicted_label"] == "A"


def test_majority_vote():
    neighbors = [
        {"label": "A", "similarity": 0.9},
        {"label": "B", "similarity": 0.8},
        {"label": "B", "similarity": 0.7},
    ]
    result = majority_vote(neighbors)
    assert result["predicted_label"] == "B"
