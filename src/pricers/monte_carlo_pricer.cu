#include "device_option_params.h"
#include "monte_carlo_pricer.h"
#include "persistent_monte_carlo.cuh"
#include "thrust/device_vector.h"

inline void check_cuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

MonteCarloPricer::MonteCarloPricer(
    std::shared_ptr<MarketDataSource> data_source, int grid_size)
    : data_source_(std::move(data_source)), grid_size_(grid_size) {}

MonteCarloPricer::Result MonteCarloPricer::price(const PricingInput& input,
                                                 std::size_t paths) const {
    if (paths == 0) {
        return Result{0.f, 0.f, 0.f};
    }

    auto history = data_source_->fetch_price_history(input.symbol, 64);
    float spot = history.empty() ? input.fallback_spot : history.back();

    DeviceOptionParams params{spot, input.strike, input.rate, input.volatility,
                              input.maturity};

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