#include <stdio.h>
#include <stdlib.h>

int dpu_send_buffer(float* tensor, int size, char* host_adress);

float* host_recv_buffer();