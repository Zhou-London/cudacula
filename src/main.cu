#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <memory>
#include <string>

#include "market.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

namespace {

struct DeviceOptionParams {
    float spot;
    float strike;
    float rate;
    float volatility;
    float maturity;
};

struct PathStats {
    float price;
    float delta;
    float vega;

    __host__ __device__ PathStats(float p = 0.f, float d = 0.f, float v = 0.f)
        : price(p), delta(d), vega(v) {}

    __host__ __device__ PathStats& operator+=(const PathStats& other) {
        price += other.price;
        delta += other.delta;
        vega += other.vega;
        return *this;
    }
};

__host__ __device__ PathStats operator+(PathStats lhs, const PathStats& rhs) {
    lhs += rhs;
    return lhs;
}

__device__ PathStats warp_reduce(PathStats value) {
    unsigned mask = 0xffffffffu;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value.price += __shfl_down_sync(mask, value.price, offset);
        value.delta += __shfl_down_sync(mask, value.delta, offset);
        value.vega += __shfl_down_sync(mask, value.vega, offset);
    }
    return value;
}

inline void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

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

}  // namespace

class MonteCarloPricer {
   public:
    struct PricingInput {
        std::string symbol;
        float fallback_spot;
        float strike;
        float rate;
        float volatility;
        float maturity;
    };

    struct Result {
        float price;
        float delta;
        float vega;
    };

    explicit MonteCarloPricer(std::shared_ptr<MarketDataSource> data_source,
                              int grid_size = 80)
        : data_source_(std::move(data_source)), grid_size_(grid_size) {}

    Result price(const PricingInput& input, std::size_t paths) const {
        if (paths == 0) {
            return Result{0.f, 0.f, 0.f};
        }

        auto history = data_source_->fetch_price_history(input.symbol, 64);
        float spot = history.empty() ? input.fallback_spot : history.back();

        DeviceOptionParams params{spot, input.strike, input.rate,
                                  input.volatility, input.maturity};

        thrust::device_vector<float> device_results(3, 0.0f);
        thrust::device_vector<int> counter(1, 0);

        float sqrt_maturity = std::sqrt(params.maturity);
        unsigned long long seed = 123456789ull;

        constexpr int kBlockSize = 256;
        persistent_monte_carlo<kBlockSize><<<grid_size_, kBlockSize>>>(
            params, static_cast<int>(paths), sqrt_maturity, seed,
            thrust::raw_pointer_cast(counter.data()),
            thrust::raw_pointer_cast(device_results.data()));

        check_cuda(cudaGetLastError());
        check_cuda(cudaDeviceSynchronize());

        thrust::host_vector<float> host_results = device_results;
        float inv_paths = 1.0f / static_cast<float>(paths);

        Result result{};
        result.price = host_results[0] * inv_paths;
        result.delta = host_results[1] * inv_paths;
        result.vega = host_results[2] * inv_paths;
        return result;
    }

   private:
    std::shared_ptr<MarketDataSource> data_source_;
    int grid_size_;
};

int main() {
    auto data_source = std::make_shared<DummyMarketDataSource>();
    MonteCarloPricer pricer(data_source);

    MonteCarloPricer::PricingInput input{"DUMMY", 100.0f, 100.0f,
                                         0.01f,   0.2f,   1.0f};
    std::size_t paths = 1 << 20;

    MonteCarloPricer::Result result = pricer.price(input, paths);

    std::cout << "Monte Carlo price: " << result.price << '\n';
    std::cout << "Delta: " << result.delta << '\n';
    std::cout << "Vega: " << result.vega << '\n';

    return 0;
}
