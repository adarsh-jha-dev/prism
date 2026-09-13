"""Key generation, hashing and verification. No database, no network."""

import pytest

from prism.auth import (
    KEY_PREFIX_LENGTH,
    GeneratedKey,
    KeyNotFound,
    ResolvedKey,
    Scope,
    generate_key,
    hash_key,
    prefix_of,
    resolve_key,
    verify_key,
)
from prism.core.ids import uuid7


class TestGeneration:
    def test_issues_a_greppable_key(self) -> None:
        assert generate_key().plaintext.startswith("prism_ak_")

    def test_every_key_is_distinct(self) -> None:
        keys = {generate_key().plaintext for _ in range(500)}
        assert len(keys) == 500

    def test_carries_enough_entropy(self) -> None:
        # token_urlsafe(32) is 43 characters of url-safe base64.
        assert len(generate_key().plaintext) == len("prism_ak_") + 43

    def test_the_prefix_is_the_head_of_the_key(self) -> None:
        key = generate_key()
        assert key.prefix == key.plaintext[:KEY_PREFIX_LENGTH]
        assert len(key.prefix) == KEY_PREFIX_LENGTH

    def test_the_stored_fields_do_not_contain_the_plaintext(self) -> None:
        """The prefix is a lookup handle; the rest of the key must not survive."""
        key = generate_key()
        body = key.plaintext[KEY_PREFIX_LENGTH:]
        assert body not in key.key_hash
        assert body not in key.prefix

    def test_the_hash_matches_the_plaintext(self) -> None:
        key = generate_key()
        assert key.key_hash == hash_key(key.plaintext)


class TestHashing:
    def test_is_deterministic(self) -> None:
        # key_hash is UNIQUE, so the digest cannot be salted.
        assert hash_key("prism_ak_abc") == hash_key("prism_ak_abc")

    def test_differs_between_keys(self) -> None:
        assert hash_key("prism_ak_abc") != hash_key("prism_ak_abd")

    def test_is_a_sha256_hex_digest(self) -> None:
        digest = hash_key("prism_ak_abc")
        assert len(digest) == 64
        assert set(digest) <= set("0123456789abcdef")


class TestVerification:
    def test_accepts_the_right_key(self) -> None:
        key = generate_key()
        assert verify_key(key.plaintext, key.key_hash) is True

    def test_rejects_another_key(self) -> None:
        assert verify_key(generate_key().plaintext, generate_key().key_hash) is False

    @pytest.mark.parametrize("wrong", ["", "prism_ak_", "not-a-key"])
    def test_rejects_junk(self, wrong: str) -> None:
        assert verify_key(wrong, generate_key().key_hash) is False

    def test_rejects_a_key_that_shares_its_prefix(self) -> None:
        """A prefix collision must not authenticate; only the digest decides."""
        a, b = generate_key(), generate_key()
        shared = GeneratedKey(plaintext=a.plaintext, prefix=b.prefix, key_hash=b.key_hash)
        assert verify_key(shared.plaintext, shared.key_hash) is False


class TestPrefix:
    def test_is_stable_for_one_key(self) -> None:
        key = generate_key()
        assert prefix_of(key.plaintext) == prefix_of(key.plaintext)

    def test_is_shared_by_every_key(self) -> None:
        assert generate_key().prefix.startswith("prism_ak_")

    def test_is_not_enough_to_authenticate(self) -> None:
        key = generate_key()
        assert verify_key(key.prefix, key.key_hash) is False


class TestScopes:
    def test_read_does_not_imply_ingest(self) -> None:
        resolved = _resolved({Scope.READ})
        assert resolved.permits(Scope.READ)
        assert not resolved.permits(Scope.INGEST)
        assert not resolved.permits(Scope.ADMIN)

    def test_admin_implies_the_others(self) -> None:
        resolved = _resolved({Scope.ADMIN})
        assert resolved.permits(Scope.READ)
        assert resolved.permits(Scope.INGEST)
        assert resolved.permits(Scope.ADMIN)

    def test_scopes_combine(self) -> None:
        resolved = _resolved({Scope.READ, Scope.INGEST})
        assert resolved.permits(Scope.INGEST)
        assert not resolved.permits(Scope.ADMIN)

    def test_no_scope_permits_nothing(self) -> None:
        resolved = _resolved(set())
        assert not any(resolved.permits(s) for s in Scope)


class TestResolution:
    async def test_a_foreign_key_never_reaches_the_database(self) -> None:
        """Rejected on format, so a scan of other systems' keys costs no query."""
        with pytest.raises(KeyNotFound):
            await resolve_key("sk-somebody-elses-key", engine=None)


def _resolved(scopes: set[Scope]) -> ResolvedKey:
    return ResolvedKey(id=uuid7(), tenant_id=uuid7(), scopes=frozenset(scopes), rate_limit_rpm=60)
