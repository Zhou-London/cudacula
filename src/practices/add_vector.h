#pragma once

namespace Practices {

void add_vector(int N, int threadsPerBlock, const float* h_A, const float* h_B,
                const float* h_C);

}  // namespace Practices