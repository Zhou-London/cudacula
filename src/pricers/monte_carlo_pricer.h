#pragma once

#include <memory>
#include <string>

#include "market_data_source.h"

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
                              int grid_size = 80);

    Result price(const PricingInput& input, std::size_t paths) const;

   private:
    std::shared_ptr<MarketDataSource> data_source_;
    int grid_size_;
};