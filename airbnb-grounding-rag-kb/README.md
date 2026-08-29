# Airbnb Grounding Verification RAG Knowledge Base

This is the DOCUMENTS layer for a Responsible AI grounding-verification prototype.

The corpus uses official Airbnb Help Center sources. `raw/` preserves source provenance records and `cleaned/` contains normalized Markdown suitable for RAG ingestion.

Recommended ingestion:
1. Ingest `cleaned/`.
2. Chunk by Markdown headings/paragraphs at roughly 400–800 tokens with modest overlap.
3. Preserve document_id, category, product, audience, region, source_url and retrieved_at.
4. Use metadata filters for home vs service/experience, India vs global, and cancellation/refund/exception categories.
5. Store source_url with each chunk for provenance.

Evaluation:
50 adversarial cases are provided with ALLOW/BLOCK/FLAG verdicts. FLAG means reservation-specific or other evidence is insufficient; BLOCK means the claim is contradicted or materially unsupported; ALLOW means the corpus contains sufficient supporting evidence.

Important: the cleaned documents are normalized policy representations, not verbatim copies. Refresh the official URLs before production or final submission because policies can change.
