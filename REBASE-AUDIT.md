# REBASE-AUDIT — rebase-2026-10-13 → upstream/main (1.106.0)

- Previous fork tip: `rebase-2026-09-12` = `e9ac4587f7`(冻结,回滚参照)
- Previous base: `362033cb2a` = upstream 1.102.0 (2026-09-10)
- New rebase target: `upstream/main` = `7af7919200` = **1.106.0**(2026-10-13 拉取)
- New branch: `rebase-2026-10-13`
- Fork commits carried over: 24(上轮 25 个中 pgbouncer bump 上游已收编,自动丢弃)
- Prior round's audit: branch `rebase-2026-09-12`(冻结)— `REBASE-AUDIT.md` @ `4595f0035e`

## What changed upstream that touches fork code (362033cb2a..7af7919200, 60 commits)

1. **测试目录大重组**(1dbde0845e 等):`tests/search_tests/` 删除、
   `tests/test_litellm/` → `tests/unit/` + `tests/integration/`。我方测试文件经
   rename 侦测随迁;IQS 测试落到新布局 `tests/unit/llms/aliyun_iqs/`。
2. **包改名**:`litellm/llms/anthropic/experimental_pass_through/` →
   `pass_through/`。fork 内引用全部随迁;**ops 仓库两处跟改**:
   `litellm-gateway/startup_patch.py`(bridge utils 导入)与
   `tests/test_effort_max_passthrough.py`。
3. **/model/info 读侧重写**(proxy_server.py `_enrich_model_info_with_litellm_data`):
   新增 discovered_model_info(进程内注册)叠加 + `_pricing_override_stamps` +
   unpriced None 置空;`_get_proxy_model_info` 改为委托一行 + 新增
   `_model_info_json_response`。能力键门禁重新挂进新结构:fill 走
   `_overlay_litellm_model_info`(db 行跳过能力键),None 回填分支补能力键守卫,
   unpriced/stamps 语义逐字保留。
4. **update_db_model model_info 合并重写**(model_management_endpoints.py):新增
   `echoed_cost_map_fields` / `without_server_derived_pricing` 回声剔除。能力键
   替换语义(未携带即清空)追加在回声剔除之后,互不干扰。
5. **messages 路由调用点新增 compaction 守卫**(`compaction is None and`):
   保留,与我方 per-deployment api_base/model_info 实参并列合入。
6. **pgbouncer 1.25.2→1.26.0 上游已做**(同 sha256)→ 我方 e9ac4587f7 rebase 自动丢弃。
7. passthrough handler 新增 `_JSON_OBJECT` TypeAdapter(纯增量,与未定价守卫无交集);
   router.py 新增 `EncryptedContentAffinityCheck` pre-call check(共存)。
8. 依赖面:venv 需 `uv sync` 重刷(mcp 新版带 CacheableResult 等);tests/unit
   conftest 新增 pytest-socket 依赖;prisma 需手动补装(同前轮,test-env only)。
9. **ConfigRepository 删掉 `CONFIG_PARAMS` 白名单**(类仍在
   `litellm/repositories/config_repository.py`,`get_all_params` 改走 find_many):
   /config/get 与 /config/param 的 param_name 门禁失去判定表 → 生产 500
   AttributeError(litellm-tf plan 读 router_settings 触发,release 验收实证)。
   fork 回补类属性(带 custom-aigw 注释)+ 新增
   `tests/unit/repositories/test_config_repository_params.py` 钉死。

## Patch disposition @1.106.0

| # | Patch | Status @1.106 | Action |
|---|---|---|---|
| 1 | 428 cooldown | NOT-SOLVED | ported as-is |
| 2 | /reset + /circuit/reset | NOT-SOLVED | ported as-is |
| 3 | GLM tool_use sanitizer | n/a(ops 仓库持有) | stays out of fork |
| 4 | websearch enabled_models gate | NOT-SOLVED | ported as-is |
| 5 | websearch native-block short-circuit | SOLVED upstream | stays dropped |
| 6 | Aliyun IQS provider | NOT-SOLVED | ported(测试随迁新布局) |
| 7 | Kimi quota_sync | NOT-SOLVED | ported as-is(含 canary revert 对) |
| 8 | periodic-quota 403/marker cooldown | NOT-SOLVED | ported as-is(6 marker:weekly×2 + entitlement×2) |
| 9 | per-deployment responses routing + K3 translation | NOT-SOLVED | ported + **compaction 守卫合入**(变更 5) |
| 10 | reasoning_effort normalize | SOLVED upstream | n/a |
| 11 | cost-map key normalization | NOT-SOLVED | ported as-is |
| 12 | passthrough unpriced logging | NOT-SOLVED(上游重写只动旁路) | ported(常量区 @both) |
| 13 | /config/get + /config/param | NOT-SOLVED | ported as-is |
| 14 | /router/settings traceback | NOT-SOLVED | ported as-is |
| 15 | ConstraintCapabilityCheck + effort 提取 | NOT-SOLVED | ported(import 冲突 @both) |
| 16 | capability keys operator 权威 | NOT-SOLVED | ported + **适配变更 3/4**(见上) |
| 17 | CI workflows | n/a | ported(schedule 删除重放) |
| 18 | pgbouncer 1.26.0 | **SOLVED upstream** | **dropped**(patch already upstream) |

**Retired this round:** pgbouncer bump(#18,上游收编)。
**Adapted:** #9(compaction 守卫)、#16(读侧/写侧上游重写后重挂能力键语义)。
**Rebase 修复(已 autosquash 回各属补丁):** router.py import @both 语法修复(96d3f126ef 系);
fork 测试 experimental_pass_through → pass_through 导入随迁(a1f9ba0f02 系)。

## Verification (2026-10-13, 本地 .venv,uv sync --frozen + proxy extras + prisma==0.11.0 + pytest-socket)

- [x] 核心套件 271 passed:test_periodic_quota_cooldown / test_quota_sync /
  test_register_model_custom_pricing / test_router_model_cost_isolation /
  test_aliyun_iqs_search / test_websearch_enabled_models /
  test_constraint_capability_check / test_capability_model_info_keys
- [x] anthropic pass_through 全家桶 + passthrough logging:1253 passed
  (含 K3 thinking 翻译 3 测试、per-deployment 路由 10 测试)
- [x] ops 全量回归(PYTHONPATH=fork):235 passed / 44 skipped(live 门控)
- [ ] Live gateway(PRE/release 验收时复核):(a) /model/info 能力键门禁在新读侧
  结构下仍生效(db 行不被 catalog/discovered 叠加);(b) compaction 守卫不影响
  k3 路由;(c) 其余沿用前轮清单(/reset、marker 冷却、responses 路由优先级)。

## Deploy

Follow AGENTS.md 发布流水线: build tag `vYYYYMMDD-hhmm` from this branch →
`deploy-image.sh <tag> --env pre` → PRE 验收 → `deploy-image.sh <tag>`(release
蓝绿)。
