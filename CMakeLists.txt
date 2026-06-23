cmake_minimum_required(VERSION 3.20)

project(reaction_diffusion_mpi VERSION 1.0.0 LANGUAGES CXX)

# Stage 3 enables aggressive CPU optimization and OpenMP.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export clangd database" FORCE)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED COMPONENTS CXX)

add_executable(reaction-diffusion src/main.cpp)
target_compile_features(reaction-diffusion PRIVATE cxx_std_17)
target_link_libraries(reaction-diffusion PRIVATE
  MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_compile_definitions(reaction-diffusion PRIVATE RD_STAGE_NAME="openmp_o3")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(reaction-diffusion PRIVATE
    -O3 -march=native -ffast-math -g -fno-omit-frame-pointer
    -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
  add_executable(reaction-diffusion-smoke src/main.cpp)
  target_compile_features(reaction-diffusion-smoke PRIVATE cxx_std_17)
  target_link_libraries(reaction-diffusion-smoke PRIVATE
    MPI::MPI_CXX OpenMP::OpenMP_CXX)
  target_compile_definitions(reaction-diffusion-smoke PRIVATE
    RD_SMOKE_TEST=1 RD_STAGE_NAME="openmp_o3")
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(reaction-diffusion-smoke PRIVATE
      -O3 -march=native -ffast-math -g -fno-omit-frame-pointer
      -Wall -Wextra -Wpedantic)
  endif()

  add_test(
    NAME reaction_diffusion_smoke
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
            $<TARGET_FILE:reaction-diffusion-smoke>)
  set_tests_properties(reaction_diffusion_smoke PROPERTIES
    PASS_REGULAR_EXPRESSION
      "stage=openmp_o3.*ranks=2.*threads_per_rank=2.*resolution=128x128.*steps=80.*output=reaction_diffusion.bmp"
    ENVIRONMENT "OMP_NUM_THREADS=2"
    TIMEOUT 30)

  add_test(
    NAME reject_program_arguments
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1
            $<TARGET_FILE:reaction-diffusion-smoke> --size 32)
  set_tests_properties(reject_program_arguments PROPERTIES WILL_FAIL TRUE)
endif()
