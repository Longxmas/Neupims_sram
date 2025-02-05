#include "AddressAllocator.h"

WgtAlloc::WgtAlloc() : _base_addr(0), _top_addr(0) {}

addr_type WgtAlloc::allocate(uint64_t size) {
    spdlog::info("weightAlloc base_addr : {}", size);
    addr_type unit = Config::global_config.dram_req_size * Config::global_config.dram_channels;
    addr_type result = _top_addr;
    _top_addr += (size + unit - 1) / unit; // 权重分配到每个channel里，激活与kv cacche由各自的channel单独保存
    // if (_top_addr & (AddressConfig::alignment - 1)) {
    //     _top_addr += AddressConfig::alignment - (_top_addr & (AddressConfig::alignment - 1));
    // }

    return result;
}

addr_type WgtAlloc::get_next_aligned_addr() {
    spdlog::info("weightAlloc base_addr : {}", _base_addr);
    ast(_top_addr > 0);
    return AddressConfig::align(_top_addr) + AddressConfig::alignment;
}