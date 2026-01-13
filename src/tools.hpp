#ifndef SRC_TOOLS_HPP
#define SRC_TOOLS_HPP

#include <iostream>

// typedef unsigned short ushort;  // 16 bit
// typedef unsigned int uint; // 32 bit

namespace trtutils
{
    template<typename T>
    std::vector<size_t> argmax_idx(const std::vector<T> v)
    {
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&v](size_t i, size_t j){return v[i] > v[j];});

        return idx;
    }

    float half_to_float(const ushort x);
    ushort float_to_half(const float x); 
}

#endif