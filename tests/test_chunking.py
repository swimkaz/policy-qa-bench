import pytest

from policyqa.chunking import chunk_markdown, slugify

DOC = "# Title\n\nFront matter.\n\n## First Part\nAlpha beta.\n\n## Second (Odd) Part\n" + "word " * 50


def test_slugify():
    assert slugify("Simplified (No-Exam) Underwriting") == "simplified-no-exam-underwriting"


def test_sections_become_chunks_and_front_matter_is_dropped():
    chunks = chunk_markdown("doc", DOC)
    assert [c.section_id for c in chunks] == ["doc#first-part", "doc#second-odd-part"]
    assert chunks[0].text == "Title | First Part\nAlpha beta."
    assert all("Front matter" not in c.text for c in chunks)


def test_long_sections_split_with_overlap():
    chunks = chunk_markdown("doc", "## Long\n" + " ".join(f"w{i}" for i in range(100)), max_words=40, overlap=10)
    assert [c.id for c in chunks] == ["doc#long@0", "doc#long@1", "doc#long@2"]
    assert {c.section_id for c in chunks} == {"doc#long"}
    assert chunks[0].body.split()[-10:] == chunks[1].body.split()[:10]
    assert chunks[-1].body.split()[-1] == "w99"


def test_overlap_must_be_smaller_than_window():
    with pytest.raises(ValueError):
        chunk_markdown("doc", DOC, max_words=10, overlap=10)


def test_corpus_section_ids_are_unique(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)) == 40
