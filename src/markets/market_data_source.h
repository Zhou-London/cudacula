#pragma once

#include "thrust/host_vector.h"

class MarketDataSource {
   public:
    virtual ~MarketDataSource() = default;
    virtual thrust::host_vector<float> fetch_price_history(
        const std::string& symbol, std::size_t count) const = 0;
};