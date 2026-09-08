# Provider fixtures

Recorded request/response cassettes for the paid providers (Ollama Cloud,
Gemini, OpenAI).

**CI must never make a paid call.** Every test that touches a paid provider
replays from here. There are no API keys in CI.

Recording (Phase 3, local only, with real keys in `.env`):

    make record-fixtures

Rules:
- Scrub keys, tokens and tenant identifiers before committing.
- Cassettes are committed — they are the contract under test.
- Re-record deliberately, and treat a diff here as an API change worth reading.
