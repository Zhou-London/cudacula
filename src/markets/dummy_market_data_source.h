#pragma once

#include <string>

#include "market_data_source.h"

class DummyMarketDataSource final : public MarketDataSource {
   public:
    thrust::host_vector<float> fetch_price_history(
        const std::string& symbol, std::size_t count) const override;
};