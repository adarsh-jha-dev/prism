"use client";

// Client component: the fetch runs in the browser, over the same CORS path the
// dashboard will use.

import { useCallback, useEffect, useState } from "react";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const DEPENDENCIES = ["postgres", "redis", "ollama"] as const;
type Dependency = (typeof DEPENDENCIES)[number];

type Check = { ok: boolean; detail?: string; error?: string };
type DepsResponse = { ok: boolean; checks: Record<Dependency, Check> };

type State =
  | { kind: "loading" }
  | { kind: "loaded"; data: DepsResponse }
  | { kind: "unreachable"; error: string };

const LABELS: Record<Dependency, string> = {
  postgres: "Postgres",
  redis: "Redis",
  ollama: "Ollama",
};

export default function StatusPage() {
  const [state, setState] = useState<State>({ kind: "loading" });
  const [checkedAt, setCheckedAt] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setState({ kind: "loading" });
    try {
      // 503 carries the per-dependency detail, so only a transport error is
      // "unreachable".
      const response = await fetch(`${API_BASE_URL}/health/deps`, {
        cache: "no-store",
      });
      setState({ kind: "loaded", data: (await response.json()) as DepsResponse });
    } catch (error) {
      setState({
        kind: "unreachable",
        error: error instanceof Error ? error.message : String(error),
      });
    }
    setCheckedAt(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main
      style={{
        maxWidth: "36rem",
        margin: "0 auto",
        padding: "4rem 1.5rem",
      }}
    >
      <h1 style={{ fontSize: "1.5rem", margin: 0 }}>Prism</h1>
      <p style={{ color: "var(--muted)", marginTop: "0.25rem" }}>
        Phase 0 — the stack is wired up when all three are green.
      </p>

      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: "2rem 0 0",
          border: "1px solid var(--border)",
          borderRadius: "0.5rem",
        }}
      >
        {DEPENDENCIES.map((name, index) => (
          <li
            key={name}
            style={{
              display: "flex",
              alignItems: "baseline",
              gap: "0.75rem",
              padding: "0.875rem 1rem",
              borderTop: index === 0 ? "none" : "1px solid var(--border)",
            }}
          >
            <StatusDot state={state} name={name} />
            <span style={{ fontWeight: 500, minWidth: "5.5rem" }}>
              {LABELS[name]}
            </span>
            <span
              style={{
                color: "var(--muted)",
                fontSize: "0.875rem",
                fontFamily: "ui-monospace, SFMono-Regular, monospace",
              }}
            >
              {describe(state, name)}
            </span>
          </li>
        ))}
      </ul>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          marginTop: "1.25rem",
        }}
      >
        <button
          onClick={() => void refresh()}
          disabled={state.kind === "loading"}
          style={{
            padding: "0.4rem 0.9rem",
            borderRadius: "0.375rem",
            border: "1px solid var(--border)",
            background: "transparent",
            color: "inherit",
            font: "inherit",
            cursor: state.kind === "loading" ? "default" : "pointer",
          }}
        >
          {state.kind === "loading" ? "Checking…" : "Re-check"}
        </button>
        {checkedAt && (
          <span style={{ color: "var(--muted)", fontSize: "0.8125rem" }}>
            last checked {checkedAt}
          </span>
        )}
      </div>

      {state.kind === "unreachable" && (
        <p style={{ color: "var(--down)", fontSize: "0.875rem" }}>
          Could not reach the API at {API_BASE_URL} — {state.error}
        </p>
      )}
    </main>
  );
}

function checkFor(state: State, name: Dependency): Check | null {
  return state.kind === "loaded" ? (state.data.checks[name] ?? null) : null;
}

function StatusDot({ state, name }: { state: State; name: Dependency }) {
  const check = checkFor(state, name);
  const color =
    check === null ? "var(--unknown)" : check.ok ? "var(--ok)" : "var(--down)";
  return (
    <span
      aria-hidden
      style={{
        width: "0.625rem",
        height: "0.625rem",
        borderRadius: "50%",
        background: color,
        flexShrink: 0,
      }}
    />
  );
}

function describe(state: State, name: Dependency): string {
  if (state.kind === "loading") return "checking…";
  if (state.kind === "unreachable") return "unknown — API unreachable";
  const check = checkFor(state, name);
  if (check === null) return "not reported";
  return check.ok ? (check.detail ?? "ok") : (check.error ?? "down");
}
