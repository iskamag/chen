#ifndef CHEN_NEURAL_H
#define CHEN_NEURAL_H
#include "chen_external.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ChenNeuralContext ChenNeuralContext;
uint32_t chen_neural_abi_version(void);
const char *chen_neural_torch_version(void);
/* Load only trusted, generated, version-pinned TorchScript archives. The caller
 * verifies its SHA256 and parsed manifest/schema/limits before invoking create.
 * Embedded chen_manifest.json must equal expected_manifest byte-for-byte.
 * File <=64MiB, manifest <=64KiB; max_batch checked against the parsed manifest.
 * Module is frozen inference-only: [B,18,8,8] float32 -> [B] int32 STM CP.
 * Create before loading/creating rules context so bundled Torch HIP binds first.
 * All five runtime identity tokens must then match the rules library. */
ChenNeuralContext *chen_neural_create(const void *archive, uint64_t archive_bytes,
    const char *expected_manifest, uint64_t manifest_bytes, int32_t device,
    uint64_t max_batch, char *error, uint64_t error_capacity);
void chen_neural_destroy(ChenNeuralContext *);
uint64_t chen_neural_runtime_identity(uint32_t selector);
/* Additive execution ABI1. Configure before first nonzero evaluation only.
 * Default mode0 is original eager JIT. Mode1 caches exact-count HIP graphs;
 * unsupported capture/budget failures are errors, without fallback or padding.
 * max_cached_shapes1..16; retained_budget_bytes bounds admitted graph pools,
 * including allocator segment rounding. It excludes transient capture memory,
 * staging/features/model tensors and other runtime allocations. Only trusted
 * generated deterministic frozen StaticEvaluator graphs are supported, with
 * native allocator/default capture reuse/fixed segments. Plugin GPU operations
 * serialize process-wide; capture requires no concurrent same-process GPU
 * activity outside this plugin, including Torch or resident rules operations.
 * Actual capture faults poison ALL inference/create operations until process
 * restart. Unprovable retirement fails the process with exit70 before freeing
 * potentially live graph/tensor/pool addresses. No automatic recovery claim. */
typedef struct ChenNeuralExecutionConfig {
    uint32_t abi_version, mode, max_cached_shapes, reserved;
    uint64_t retained_budget_bytes;
} ChenNeuralExecutionConfig; /* size24 */
typedef struct ChenNeuralExecutionStats {
    uint32_t abi_version, mode, max_cached_shapes, first_used;
    uint64_t cached_shapes, hits, misses, captures, evictions;
    uint64_t retained_pool_bytes, staging_logical_bytes, peak_observed_pool_bytes;
    double capture_wall_ms;
} ChenNeuralExecutionStats; /* size88 */
uint32_t chen_neural_execution_abi_version(void);
int32_t chen_neural_set_execution(ChenNeuralContext *, const ChenNeuralExecutionConfig *,
    char *, uint64_t);
int32_t chen_neural_execution_stats(const ChenNeuralContext *, ChenNeuralExecutionStats *,
    char *, uint64_t);
/* Diagnostic failure injection is enabled only in CHEN_NEURAL_TESTING builds;
 * ordinary libraries always reject. Stages1poolbegin2capturebegin3graphend
 * 4admission5unproven-retirement-failstop; one shot, stage0 clears.
 * Not an execution policy. */
int32_t chen_neural_test_inject_failure(ChenNeuralContext *, uint32_t stage);
/* Synchronous ChenExternalEvaluate implementation; borrowed device pointers
 * and exclusive context ownership. Dynamic count0 succeeds without dispatch.
 * Decodes canonical schema1 features entirely on GPU, executes same-model JIT,
 * validates dtype/shape/device/range, copies scores D2D and drains on ALL paths.
 * CPU scalar validation/telemetry is permitted, board payload download is not. */
int32_t chen_neural_evaluate(void *, int32_t, const ChenRulePosition *, uint64_t,
    int32_t *, char *, uint64_t);
#ifdef __cplusplus
}
#endif
#endif
