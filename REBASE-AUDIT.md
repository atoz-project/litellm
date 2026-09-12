# REBASE-AUDIT — rebase-2026-09 → upstream/main (2026-09-12)

- Previous fork tip: `rebase-2026-09` = `f7c75d17c4` (frozen, rollback reference)
- Previous base: `7d5b6456ba` = upstream 1.101.0 (2026-09-03)
- New rebase target: `upstream/main` = `362033cb2a` = **1.102.0** (2026-09-10)
- New branch: `rebase-2026-09-12`
- Fork commits carried over: 15 patches + 2 new fixes (see below)
- Prior round's audit: branch `rebase-2026-09` (frozen) — `REBASE-AUDIT.md` @ `16da628b78`

## What changed upstream that touches fork code (7d5b6456ba..362033cb2a)

1. **Cooldown storage moved** (`cooldown_cache.py`): cooldown entries now live in
   `CooldownCache.cooldown_store` (dedicated DualCache: own in-memory tier +
   lazily-attached Redis, batch re-read interval `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS`)
   instead of the router-wide `cache`. Fork code deleting cooldown keys from
   `cooldown_cache.cache` became a silent no-op.
2. **allowed_fails counters moved** (`cooldown_handlers.py`): from
   `Router.failed_calls` (deleted upstream) into the router's shared DualCache,
   keys `deployment:{model_id}:allowed_fails[:{exception-class|generic}]`,
   fleet-wide via `DualCache.increment_cache`.
3. **Passthrough logging handler refactored** (+168/-86): costing extracted into
   `_compute_response_cost`; the *partial-stream* failure path is now natively
   graceful (`_cost_partial_stream_or_zero`). The *success* path
   (`_create_anthropic_response_logging_payload`) still wraps everything in one
   try — unpriced models still abort the whole payload there.
4. Big additive churn in `proxy_server.py` (+846), `router.py` (+806),
   `utils.py` (+281), `_types.py` (+159), `types/utils.py` (+106): no overlap
   with fork semantics beyond the above.

## Patch disposition @1.102.0

| # | Patch | Status @1.102 | Action |
|---|---|---|---|
| 1 | 428 cooldown | NOT-SOLVED (no 428 branch upstream) | ported as-is |
| 2 | /reset + /circuit/reset | NOT-SOLVED, but see upstream change 1/2 | ported + **adapted** (commit 4ecd6e4f46) |
| 3 | GLM tool_use sanitizer | n/a (ops repo owns it) | stays out of fork |
| 4 | websearch enabled_models gate | NOT-SOLVED (upstream file unchanged) | ported as-is |
| 5 | websearch native-block short-circuit | SOLVED upstream (dropped 2026-09-06) | stays dropped |
| 6 | Aliyun IQS provider | NOT-SOLVED | ported as-is |
| 7 | Kimi quota_sync | NOT-SOLVED, but see upstream change 1 | ported + **adapted** (4ecd6e4f46) |
| 8 | periodic-quota 403/marker cooldown | NOT-SOLVED (403 still never cools upstream) | ported + **marker drift fix** (0f748d9f36) |
| 9 | per-deployment responses routing + K3 translation | NOT-SOLVED (upstream still global-flag-first blanket) | ported as-is |
| 10 | reasoning_effort normalize | SOLVED upstream (was never a fork patch) | n/a |
| 11 | cost-map key normalization | NOT-SOLVED (`shared_key`/`_backend_cost_map_keys` still unconditionally prefix) | ported as-is |
| 12 | passthrough unpriced logging | **PARTIAL → slimmer**: upstream now graceful for partial-stream costing; success path unchanged | **ported into `_compute_response_cost`** — one wrapper serves both paths |
| 13 | /config/get + /config/param | NOT-SOLVED | ported as-is |
| 14 | /router/settings traceback | NOT-SOLVED (file unchanged upstream) | ported as-is |
| 15 | ConstraintCapabilityCheck + effort extraction | NOT-SOLVED (fork-only files untouched upstream) | ported as-is |
| 16 | CI workflows | n/a | ported as-is |

**Retired this round:** none newly dropped (everything still load-bearing).
**Slimmed:** #12 (half the mechanism is upstream now).
**New fixes this round:**
- `4ecd6e4f46` — /reset + quota_sync unblock adapted to cooldown_store /
  allowed_fails DualCache keying (upstream changes 1+2). Without this, /reset
  silently cleared nothing and recovered Kimi keys stayed cooled forever.
- `0f748d9f36` — Kimi weekly-quota wording drift: markers += "weekly (7-day)
  usage limit" + "membership/subscription" (live-observed 2026-09 text);
  regression test pins the verbatim production error.

## Verification (2026-09-12, local .venv py3.13, proxies unset)

- [x] test_periodic_quota_cooldown.py: 5 passed (incl. new weekly-wording case)
- [x] test_quota_sync.py: 22 passed (2 initial failures = upstream change 1, fixed in 4ecd6e4f46)
- [x] test_register_model_custom_pricing.py + test_router_model_cost_isolation.py: 123 passed
- [x] test_websearch_enabled_models.py + test_aliyun_iqs_search.py + test_constraint_capability_check.py: 80 passed
- [x] anthropic adapters + messages handler + passthrough logging: 339 passed
- [x] ops regression `test_effort_max_passthrough.py` (PYTHONPATH=../litellm): 30 passed
- [ ] **Live gateway (post-deploy gates):** (a) /reset actually clears allowed_fails
  counters + cooldowns on the new keying (probe: cool a deployment, /reset, confirm
  immediate re-pick); (b) Kimi weekly-403 marker now cools + quota_sync unblocks on
  recovery (next real wall-hit); (c) per-deployment responses-routing precedence
  unchanged for existing pools; (d) websearch native-block short-circuit via
  upstream WEBSEARCH_EMIT_NATIVE_BLOCKS_KEY (unchanged since 1.101, verified then).
- [ ] Note: `tests/test_litellm/proxy/conftest.py` requires `prisma` — installed
  `prisma==0.11.0` into the local .venv for collection (test-env only).

## Deploy

Follow AGENTS.md 发布流水线: build tag `vYYYYMMDD-hhmm` from this branch →
`deploy-image.sh <tag> --env pre` → PRE 验收 → `deploy-image.sh <tag>` (release
蓝绿). Live-gateway checklist above doubles as the 验收探针列表.
