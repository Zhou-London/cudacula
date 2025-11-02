#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cub/block/block_reduce.cuh>
#include <persistent_monte_carlo.cuh>

#include "path_stats.h"

template <int BlockSize>
__global__ void persistent_monte_carlo(DeviceOptionParams params, int num_paths,
                                       float sqrt_maturity,
                                       unsigned long long seed, int* counter,
                                       float* results) {
    using BlockReduce = cub::BlockReduce<PathStats, BlockSize>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int lane = threadIdx.x & 31;
    PathStats warp_accumulator;

    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, global_thread_id, 0, &rng_state);

    const float discount = expf(-params.rate * params.maturity);
    const float drift =
        (params.rate - 0.5f * params.volatility * params.volatility) *
        params.maturity;

    while (true) {
        int path_id = atomicAdd(counter, 1);
        if (path_id >= num_paths) {
            break;
        }

        float gaussian = curand_normal(&rng_state);
        float diffusion = params.volatility * sqrt_maturity * gaussian;
        float terminal_price = params.spot * expf(drift + diffusion);
        float payoff = fmaxf(terminal_price - params.strike, 0.0f);

        PathStats path_contrib;
        path_contrib.price = discount * payoff;

        float indicator = terminal_price > params.strike ? 1.0f : 0.0f;
        path_contrib.delta =
            discount * indicator * terminal_price / params.spot;

        float d_st_d_sigma =
            terminal_price *
            (sqrt_maturity * gaussian - params.volatility * params.maturity);
        path_contrib.vega = discount * indicator * d_st_d_sigma;

        path_contrib = warp_reduce(path_contrib);
        if (lane == 0) {
            warp_accumulator += path_contrib;
        }
    }

    PathStats thread_contrib = (lane == 0) ? warp_accumulator : PathStats{};
    PathStats block_sum = BlockReduce(temp_storage).Sum(thread_contrib);

    if (threadIdx.x == 0) {
        atomicAdd(results + 0, block_sum.price);
        atomicAdd(results + 1, block_sum.delta);
        atomicAdd(results + 2, block_sum.vega);
    }
}

template __global__ void persistent_monte_carlo<256>(
    DeviceOptionParams params, int num_paths, float sqrt_maturity,
    unsigned long long seed, int* counter, float* results);