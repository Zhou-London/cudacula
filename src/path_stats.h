#pragma once

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