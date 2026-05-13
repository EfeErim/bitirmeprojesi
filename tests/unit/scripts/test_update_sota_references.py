from scripts.update_sota_references import (
    CANDIDATE_SECTION_BEGIN,
    CANDIDATE_SECTION_END,
    render_candidate_section,
    update_guide_with_candidate_section,
)


def test_update_guide_replaces_existing_candidate_section():
    old_section = "\n".join(
        [
            CANDIDATE_SECTION_BEGIN,
            "#### Latest Automated Candidate Scan",
            "",
            "Generated: `old`",
            "",
            CANDIDATE_SECTION_END,
        ]
    )
    guide = f"# Guide\n\n{old_section}\n\n### Phase 2 Checklist"
    new_section = render_candidate_section([], "2026-05-13T18:00:00Z")

    updated = update_guide_with_candidate_section(guide, new_section)

    assert updated.count(CANDIDATE_SECTION_BEGIN) == 1
    assert updated.count(CANDIDATE_SECTION_END) == 1
    assert "Generated: `old`" not in updated
    assert "Generated: `2026-05-13T18:00:00Z`" in updated
    assert "### Phase 2 Checklist" in updated


def test_update_guide_inserts_candidate_section_before_phase_2_when_missing():
    guide = "# Guide\n\n### Phase 1 Checklist\n\n### Phase 2 Checklist\n"
    section = render_candidate_section([], "2026-05-13T18:00:00Z")

    updated = update_guide_with_candidate_section(guide, section)

    assert updated.index(CANDIDATE_SECTION_BEGIN) < updated.index("### Phase 2 Checklist")
    assert "No new papers found for configured queries." in updated


def test_render_candidate_section_reports_query_errors():
    section = render_candidate_section(
        [],
        "2026-05-13T18:00:00Z",
        query_errors=[("bioclip", "network unavailable")],
    )

    assert "Candidate scan could not query all configured sources:" in section
    assert "`bioclip`: network unavailable" in section
    assert "No new papers found for configured queries." not in section


def test_render_candidate_section_summarizes_permission_errors():
    section = render_candidate_section(
        [],
        "2026-05-13T18:00:00Z",
        query_errors=[("bioclip", "Failed to connect: [WinError 10013] permission denied")],
    )

    assert "`bioclip`: network access blocked by local permissions" in section
    assert "WinError" not in section
