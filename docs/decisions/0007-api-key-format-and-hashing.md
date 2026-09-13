# 0007 — API key format and hashing

- **Status:** accepted
- **Date:** 2026-09-13
- **Relates to:** [0006](0006-postgres-rls-for-tenant-isolation.md)

## Context

Migration 0003 gave `api_keys` a `key_prefix` for lookup and kept `key_hash`
`UNIQUE`. Both constrain what the key format and hash can be, and both are
expensive to change afterwards: the hash is one-way, so altering either the
format or the digest means every issued key stops resolving and has to be
re-issued out of band.

## Decision

**Format: `prism_ak_<43 url-safe base64 characters>`.** The body is
`secrets.token_urlsafe(32)` — 256 bits from the OS CSPRNG.

A fixed, greppable literal prefix makes a leaked key identifiable in logs and by
secret scanners, which is the main practical defence once a key is out.

**`key_prefix` is the first 16 characters**, i.e. the literal `prism_ak_` plus
seven characters of the body. It is stored in plaintext and indexed, so lookup
is an index hit instead of hashing every row.

Seven base64 characters is ~42 bits, enough that a prefix lookup almost always
returns one row, while collisions stay legal — the index is deliberately not
unique and the hash comparison is what actually identifies the key.

**Hash: unsalted SHA-256, hex.** Verification is `hmac.compare_digest`.

Not bcrypt, argon2 or scrypt. Those exist to make brute force expensive against
*low-entropy* secrets that humans chose. This secret has 256 bits of CSPRNG
entropy, so there is no dictionary to run and no offline attack that a slow KDF
would meaningfully hinder. Against that, a deliberately slow hash would run on
every authenticated request inside a 6s end-to-end latency budget.

It also has to be deterministic: `key_hash` is `UNIQUE`, and a per-row salt
would make identical keys hash differently and defeat both that constraint and
the lookup.

**The plaintext is returned exactly once, at generation, and never stored.**
`generate_key()` returns it in memory; only the prefix and the digest are
persisted. There is no recovery path, by design.

## Consequences

- A lost key is re-issued, never recovered.
- Changing `PREFIX_LENGTH`, the literal prefix or the digest invalidates every
  key in the table. They are module constants in `prism.auth`, not `Settings`
  values, because they are not tunable — a deployment that changed one would
  silently stop authenticating.
- Key lookup is two steps: an indexed prefix query returning candidates, then a
  constant-time digest comparison. Callers must not short-circuit the second
  step even when the first returns exactly one row.
- Revocation and expiry are checked at resolution, not at lookup, so a revoked
  key is a distinguishable outcome rather than a missing row.
- SHA-256 of a leaked database still exposes nothing usable: reversing it means
  brute-forcing 256 bits. The threat this design does not address is a key
  leaked in plaintext, which is what the greppable prefix is for.
