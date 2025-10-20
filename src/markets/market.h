#pragma once

#include <string>
#include "thrust/host_vector.h"

class MarketDataSource {
   public:
    virtual ~MarketDataSource() = default;
    virtual thrust::host_vector<float> fetch_price_history(
        const std::string& symbol, std::size_t count) const = 0;
};

class DummyMarketDataSource final : public MarketDataSource {
   public:
    thrust::host_vector<float> fetch_price_history(
        const std::string& symbol, std::size_t count) const override {
        thrust::host_vector<float> prices(count);
        float base_price = 100.0f + static_cast<float>(symbol.size());
        for (std::size_t i = 0; i < count; ++i) {
            prices[i] = base_price + static_cast<float>(i) * 0.5f;
        }
        return prices;
    }
};