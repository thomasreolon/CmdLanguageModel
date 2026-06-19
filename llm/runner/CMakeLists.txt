cmake_minimum_required(VERSION 3.16)
project(qq_llm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Expect llama.cpp checked out at ../llama.cpp (git submodule). It builds the `llama`
# target we link against. Phase 2 adds the `denselogic` arch inside that same tree.
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH "path to llama.cpp")

if(EXISTS "${LLAMA_DIR}/CMakeLists.txt")
  add_subdirectory(${LLAMA_DIR} llama_build EXCLUDE_FROM_ALL)
  add_executable(qq-llm qq-llm.cpp)
  target_link_libraries(qq-llm PRIVATE llama)
  # so the binary can find tokenizer_spec.json / models/ relative to the repo:
  target_compile_definitions(qq-llm PRIVATE QQ_REPO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../..")
else()
  message(WARNING
    "llama.cpp not found at ${LLAMA_DIR}. "
    "Run:  git submodule add https://github.com/ggml-org/llama.cpp llm/llama.cpp")
endif()
