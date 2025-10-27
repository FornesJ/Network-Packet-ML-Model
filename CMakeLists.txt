cmake_minimum_required(VERSION 3.10)
project(doca_tcp_dma C)

set(CMAKE_C_STANDARD 11)

# Adjust DOCA path if needed:
set(DOCA_SDK_PATH /opt/mellanox/doca)

include_directories(${DOCA_SDK_PATH}/include)

set(APP_PATH /home/jorgetf/testmodel/Network-Packet-ML-Model/dma_transfer)

add_executable(dma_host ${APP_PATH}/dma_copy_host.c)
target_link_libraries(dma_host
    ${DOCA_SDK_PATH}/lib/libdoca_mmap.so
    ${DOCA_SDK_PATH}/lib/libdoca_buffer.so
    ${DOCA_SDK_PATH}/lib/libdoca_error.so
)

add_executable(dma_dpu ${APP_PATH}/dma_copy_dpu.c)
target_link_libraries(dma_dpu
    ${DOCA_SDK_PATH}/lib/libdoca_mmap.so
    ${DOCA_SDK_PATH}/lib/libdoca_buffer.so
    ${DOCA_SDK_PATH}/lib/libdoca_dev.so
    ${DOCA_SDK_PATH}/lib/libdoca_dma.so
    ${DOCA_SDK_PATH}/lib/libdoca_pe.so
    ${DOCA_SDK_PATH}/lib/libdoca_error.so
)

