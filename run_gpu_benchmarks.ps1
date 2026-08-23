# Reproduce TinyLlama Vulkan GPU benchmark sweeps on the RX 9070 XT.
# Assumes:
#   - llama.cpp Vulkan binaries are in .\llama_cpp_vulkan\
#   - GGUF models are already in .\models\gguf\
#   - results\ already exists
#
# Each command runs:
#   prompt processing: 128, 512, 1024 tokens
#   generation: 64, 128, 256 tokens
#   full GPU offload: -ngl 99
#   5 repetitions
#   CSV output

.\llama_cpp_vulkan\llama-bench.exe `
-m ".\models\gguf\tinyllama-1.1b-chat-v1.0.Q8_0.gguf" `
-p 128,512,1024 `
-n 64,128,256 `
-ngl 99 `
-r 5 `
-o csv > ".\results\tinyllama_q8_vulkan_sweep.csv"


.\llama_cpp_vulkan\llama-bench.exe `
-m ".\models\gguf\tinyllama-1.1b-chat-v1.0.Q6_K.gguf" `
-p 128,512,1024 `
-n 64,128,256 `
-ngl 99 `
-r 5 `
-o csv > ".\results\tinyllama_q6_vulkan_sweep.csv"


.\llama_cpp_vulkan\llama-bench.exe `
-m ".\models\gguf\tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf" `
-p 128,512,1024 `
-n 64,128,256 `
-ngl 99 `
-r 5 `
-o csv > ".\results\tinyllama_q5_vulkan_sweep.csv"


.\llama_cpp_vulkan\llama-bench.exe `
-m ".\models\gguf\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" `
-p 128,512,1024 `
-n 64,128,256 `
-ngl 99 `
-r 5 `
-o csv > ".\results\tinyllama_q4_vulkan_sweep.csv"