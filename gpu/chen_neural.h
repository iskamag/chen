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
