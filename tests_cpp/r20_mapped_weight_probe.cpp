// Verify file-backed vs ordinary host allocation using the actual model loader.
#include "ggml-backend.h"
#include "llama-cpp.h"
#include "llama-model.h"
#include <iostream>
#include <string>
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const bool mapped = std::string(argv[2]) == "mmap", expected = std::string(argv[3]) == "eligible";
    ggml_backend_load_all();
    llama_backend_init();
    auto params = llama_model_default_params();
    ggml_backend_dev_t cpu[] = {nullptr};
    params.devices = cpu;
    params.n_gpu_layers = 0;
    params.load_mode = mapped ? LLAMA_LOAD_MODE_MMAP : LLAMA_LOAD_MODE_NONE;
    params.lazy_mode = LLAMA_LAZY_MODE_OFF;
    llama_model_ptr model(llama_model_load_from_file(argv[1], params));
    if (!model) return 3;
    int total = 0, eligible = 0;
    for (int il = 0; il < 6; ++il) {
        for (auto op : {"gate","up","down"}) {
            const auto name = "blk." + std::to_string(il) + ".ffn_" + op + "_exps.weight";
            const auto * w = model->get_tensor(name.c_str());
            if (!w || !w->buffer || !w->data || !ggml_backend_buffer_is_host(w->buffer)) return 4;
            ++total;
            eligible += model->moe_readahead_eligible(w);
        }
    }
    std::cout << "{\"mapped\":" << mapped << ",\"host_expert_tensors\":" << total << ",\"eligible\":" << eligible << "}\n";
    return eligible == (expected ? total : 0) ? 0 : 5;
}
