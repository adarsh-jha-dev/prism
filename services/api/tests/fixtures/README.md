# Provider fixtures

Recorded request/response cassettes for the paid providers (Ollama Cloud,
Gemini, OpenAI).

**CI must never make a paid call.** Every test that touches a paid provider
replays from here. There are no API keys in CI.

Recording (local only, with real keys in `.env`):

    make record-fixtures                            # all of them
    make record-fixtures names="gemini/table_page"  # or just one

Rules:
- Scrub keys, tokens and tenant identifiers before committing. The recorder
  records no request headers at all, and refuses to write a file the key
  appears in.
- Cassettes are committed — they are the contract under test.
- Re-record deliberately, and treat a diff here as an API change worth reading.

## Shape

```json
{
  "provenance": "recorded",
  "request":  {"method": "POST", "url": "https://..."},
  "response": {"status_code": 200, "json": {...}}
}
```

`tests/cassette.py` matches on method and URL only. The request body carries a
base64 page render, and asserting on those bytes would turn every pypdfium2
upgrade into a test failure; what the request must *contain* is asserted
directly in `test_vision_gemini.py`.

## Provenance

`"provenance": "recorded"` means the exchange came off the wire.
`"synthetic"` means it was hand-authored from the provider's documented
response shape — it exercises our parsing but is not evidence about the
provider, and should be re-recorded when a key is available.

| Cassette | Provenance |
|---|---|
| `gemini/table_page.json` | recorded — a page whose table becomes one `table` chunk |
| `gemini/prose_page.json` | recorded — a flagged page the model found nothing on |

The local (`ollama`) vision lane has no cassette: it is free to call, and its
unit tests stub the transport directly. The live model is exercised by
`test_vision_ollama_live.py`, marked `ollama` and deselected in CI.
