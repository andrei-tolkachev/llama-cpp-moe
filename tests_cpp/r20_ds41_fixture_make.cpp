// Reuse the fixture metadata and seeded tensor fillers from the pinned tree.
// The dedicated Q6_K table and 256-element rows activate the R19 graph guard.
#define main upstream_arch_tool_main
#include "test-llama-archs.cpp"
#undef main

static void fill_fixture_tensor(ggml_tensor *tensor, void *user) {
    if (tensor->type != GGML_TYPE_Q6_K && tensor->type != GGML_TYPE_IQ3_XXS) {
        set_tensor_data(tensor, user);
        return;
    }

    const auto &params = *static_cast<tensor_data_params *>(user);
    std::vector<float> floats(ggml_nelements(tensor));
    fill_normal(floats, tensor->name, params);
    std::vector<uint8_t> quantized(ggml_nbytes(tensor));
    const std::vector<float> importance(tensor->ne[0], 1.0f);
    const size_t written = ggml_quantize_chunk(tensor->type, floats.data(), quantized.data(), 0,
                                               ggml_nrows(tensor), tensor->ne[0], importance.data());
    GGML_ASSERT(written == quantized.size());
    ggml_backend_tensor_set(tensor, quantized.data(), 0, quantized.size());
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    auto meta = get_gguf_ctx(LLM_ARCH_DEEPSEEK41, true);
    kv_writer kv(LLM_ARCH_DEEPSEEK41, meta.get());
    kv.add_kv(LLM_KV_CONTEXT_LENGTH, uint32_t(512));
    kv.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, uint32_t(256));
    kv.add_kv(LLM_KV_EXPERT_COUNT, uint32_t(4));
    kv.add_kv(LLM_KV_EXPERT_USED_COUNT, uint32_t(2));
    kv.add_kv(LLM_KV_EMBEDDING_LENGTH, uint32_t(256));
    kv.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, uint32_t(256));

    ggml_tensor hint{};
    hint.type = GGML_TYPE_Q6_K;
    ggml_set_name(&hint, "per_layer_token_embd.weight");
    gguf_add_tensor(meta.get(), &hint);
    hint.type = GGML_TYPE_IQ3_XXS;
    for (int layer = 0; layer < 6; ++layer) {
        for (const char *kind : {"gate", "up", "down"}) {
            ggml_format_name(&hint, "blk.%d.ffn_%s_exps.weight", layer, kind);
            gguf_add_tensor(meta.get(), &hint);
        }
    }
    auto model_params = llama_model_default_params();
    ggml_backend_dev_t devices[] = {nullptr};
    model_params.devices = devices;
    tensor_data_params filler{123, .1f};
    init_normal_pool(filler.seed, filler.stdev);
    llama_model_ptr model(llama_model_init_from_user(meta.get(), fill_fixture_tensor, &filler, model_params));
    if (!model) return 3;
    llama_model_save_to_file(model.get(), argv[1]);
    return 0;
}
