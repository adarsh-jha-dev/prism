# 0009 — API key lifecycle, and a tenant cannot lock itself out

- **Status:** accepted
- **Date:** 2026-09-13
- **Relates to:** [0007](0007-api-key-format-and-hashing.md)

## Context

`api_keys` carries `scopes`, `revoked_at` and `expires_at`, and `resolve_key`
honours all three, but nothing could set them: a tenant got one all-scopes key
at creation. `POST/GET/DELETE /keys` add issuing, listing and revoking under the
caller's own tenant.

Revocation is the one operation that can remove a tenant's ability to manage
itself, and there is no self-service recovery — `/tenants` only creates tenants.

## Decision

**A revoke is refused (409) if it would leave the tenant no admin key that is
unrevoked and has no `expires_at`.** Self-revocation is allowed when another such
key remains.

Forbidding self-revocation was rejected: it does not prevent lockout. A
short-lived admin key can revoke the permanent one, and the tenant is locked out
when it expires. It also blocks rotating a key with itself.

Counting any currently unexpired admin key as a survivor was rejected for the
same reason — it defers the lockout rather than preventing it.

**Concurrent revokes are serialized on the tenant row** (`SELECT … FOR NO KEY
UPDATE`). Without it, two admin keys revoking each other can each count the
other as the survivor. The weaker lock mode keeps foreign-key inserts
(`usage_records`, new keys) from blocking behind a revoke.

Issuing is not restricted: an admin key may issue any scope set, since admin
already implies every scope.

## Consequences

- A tenant cannot have every admin key expire. Rotating a permanent admin key
  means issuing the replacement before revoking the old one.
- Expired keys are not revoked or deleted; they stay listed with `expires_at` in
  the past.
- Tenants created before this with no permanent admin key are unaffected: the
  check only runs when the key being revoked is itself one.
