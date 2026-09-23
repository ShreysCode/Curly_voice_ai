import json

from app.knowledge.knowledge import KnowledgeBase


def test_organization_context(tmp_path):

    file = tmp_path / "data.json"

    file.write_text(
        json.dumps({
            "organization": {
                "name": "ICAR",
                "description": "Test organization"
            },
            "lab": {
                "name": "Test Lab"
            }
        })
    )

    kb = KnowledgeBase(file)

    result = kb.get_context(
        "organization"
    )

    assert "ICAR" in result
    assert "Test organization" in result


def test_lab_context(tmp_path):

    file = tmp_path / "data.json"

    file.write_text(
        json.dumps({
            "organization": {},
            "lab": {
                "name": "Test Lab",
                "opening_time": "09:00"
            }
        })
    )

    kb = KnowledgeBase(file)

    result = kb.get_context(
        "lab"
    )

    assert "Test Lab" in result
    assert "09:00" in result