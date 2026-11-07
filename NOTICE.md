# Attribution and scope

This is an independent, AI-assisted MoE patch collection for llama.cpp, not an official llama.cpp, Project Maya or Strata release. No upstream endorsement is implied.

- llama.cpp / GGML: MIT, copyright retained in LICENSE. Base commit dde879bc4f954b809892fd0d660515f42536394a.
- Project Maya: the included maya-store donor files and adapted IQ3 AVX2 routines retain their MIT notices in vendor/maya-store/LICENSE, third_party_licenses/ and tests/numerics/LICENSE-MAYA. The R21 IQ3 donor is src/ds4/ds4_iq_cpu.cpp at commit 3589d6f12a3f77e813b184b51a586f8fc3de2185; the local source file SHA-256 is 59910bfee297bce19f8edd9c7837ee133b77f6d09323e5c690a7457eff7bd121. Kernel adaptations and benchmark extraction are identified separately from original project code.
- Strata: source studied at 99866a47a2b8a21b131aad162a584ee9da92f8f0, installed engine 0.1.55. Applicable MIT notice is retained in third_party_licenses/strata-MIT.txt. The complete Strata engine is not bundled. Do not equate mechanism-inspired work with a full port of its scheduler.

The inherited R20 notice offered new experimental code, tools and documentation under the same MIT terms. This package preserves that licensing and all supplied notices; it does not claim donor code as newly authored here.

An earlier local NOTICE incorrectly said no Maya source was included while listing the vendored Maya component. This publication copy corrects that contradiction. The untouched local snapshot remains in the separate private backup.

No model weights, user conversations, service configuration, access keys or local deployment backups are part of the publication repository. Performance and quality limitations are documented, including negative results and CUDA failures. Public names omit the historical experiment identifiers.

Frequency eviction adapts Strata-style hit counts, decay and coldest-victim selection to GGML cache maps. The retained MIT notice is also supplied as third_party_licenses/LICENSE-Strata. The private backing-file release path acts only after private writable host registration; it is opt-in.

CPU gate/up/clamp fusion adapts the Strata native_gu_rows mechanism to the existing GGML custom operator and CPU pool. It shares input quantization while retaining GGML vec_dot and down projection. Only compatible single-token pre-clamp architectures opt in; other cases retain their existing path.

The phase-budget patch reuses llama.cpp model/KV/scheduler lifetimes and applies the phase-dependent VRAM budgeting idea studied in Strata. It recreates the expert cache, rather than claiming a port of Strata's in-place cache-loan mechanism. The included source-code benchmark corpus contains MIT-licensed project and donor code with notices retained in this kit.
