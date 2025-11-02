#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <memory>

#include "dummy_market_data_source.h"
#include "monte_carlo_pricer.h"

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
