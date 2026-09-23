from pathlib import Path

from app.knowledge.info_repository import InfoRepository


REPO_PATH = Path("app/knowledge/info_repo.json")


def test_nrcy_alias_is_normalized():
    repo = InfoRepository(REPO_PATH)

    text, matches = repo.normalize(
        "Where is the nrc why experimental farm?"
    )

    assert "NRCY" in text
    assert any(
        item["heard"] == "nrc why"
        and item["canonical"] == "NRCY"
        for item in matches
    )


def test_nyukmadung_variant_is_normalized():
    repo = InfoRepository(REPO_PATH)

    text, _ = repo.normalize(
        "Tell me about the nyukmadan farm."
    )

    assert "Nyukmadung" in text


def test_normal_text_is_not_changed():
    repo = InfoRepository(REPO_PATH)

    text, matches = repo.normalize(
        "What is the experimental yak farm?"
    )

    assert text == "What is the experimental yak farm?"
    assert matches == []


def test_alias_matching_is_conservative():
    repo = InfoRepository(REPO_PATH)

    text, matches = repo.normalize(
        "Please tell me about NRCY today."
    )

    assert text == "Please tell me about NRCY today."
    assert matches == []
