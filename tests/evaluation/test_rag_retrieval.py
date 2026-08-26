from app.rag.evaluation import evaluate_retrieval


def test_clean_rag_retrieval_integrity() -> None:
    result = evaluate_retrieval(k=5)
    assert result["split"] == "clean_test"
    assert result["n"] == 168
    assert result["temporal_policy"] == "serial"
    assert result["temporal_violations"] == 0
    assert result["duplicate_id_queries"] == 0
    assert result["evidence_available"] + result["zero_evidence"] == result["n"]
    assert 0.0 <= result["evidence_coverage"] <= 1.0
    assert "H(" in result["retriever"]
