#pragma once

#include "device_option_params.h"

template <int BlockSize>
__global__ void persistent_monte_carlo(DeviceOptionParams params, int num_paths,
                                       float sqrt_maturity,
                                       unsigned long long seed, int* counter,
                                       float* results);