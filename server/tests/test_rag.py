import rag


def test_rag_ranks_relevant_chunk_first():
    chunks = [
        rag.Chunk("a#0", "a.md", "Optimization",
                  "fmincon solves constrained nonlinear minimization with bounds"),
        rag.Chunk("b#0", "b.md", "Plotting", "plot draws a 2-D line graph from vectors"),
    ]
    idx = rag.RagIndex()
    idx.build(chunks)
    hits = idx.query("how do I use fmincon for constrained optimization", top_k=1)
    assert hits
    assert hits[0][1].source == "a.md"


def test_rag_query_with_no_overlap_returns_empty():
    idx = rag.RagIndex()
    idx.build([rag.Chunk("a#0", "a.md", "T", "hello world")])
    assert idx.query("zzz qqq nonexistent", top_k=3) == []


def test_chunk_markdown_splits_on_headings(tmp_path):
    p = tmp_path / "d.md"
    p.write_text("# Title\n\n## Section A\nalpha text\n\n## Section B\nbeta text\n")
    chunks = rag.chunk_markdown(p, "d.md", max_chars=4000)
    titles = [c.title for c in chunks]
    assert "Section A" in titles and "Section B" in titles
