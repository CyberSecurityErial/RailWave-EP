#pragma once

#include <cstdint>
#include <limits>

#include <deep_ep/common/compiled.cuh>
#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>

namespace deep_ep::elastic::rail_balance {

enum class Policy : int {
    Off = 0,
    Active = 1,
    All = 2,
    Incast = 3,
};

static constexpr int kNumMaxDestinations = 32;
static constexpr int kForwardSrcTokenDim = 0;
static constexpr int kForwardLastTokenDim = 1;
static constexpr int kForwardProxySlotDim = 2;
static constexpr int kForwardRouteBaseDim = 3;

__forceinline__ __device__ __host__ constexpr bool is_valid_policy(const int policy) {
    return policy >= static_cast<int>(Policy::Off) and
           policy <= static_cast<int>(Policy::Incast);
}

__forceinline__ __device__ __host__ constexpr bool is_enabled(const int policy) {
    return policy != static_cast<int>(Policy::Off);
}

__forceinline__ __device__ __host__ constexpr bool is_incast_policy(const int policy) {
    return policy == static_cast<int>(Policy::Incast);
}

__forceinline__ __device__ __host__ constexpr int get_forward_metadata_dims(const int num_topk) {
    return kForwardRouteBaseDim + 2 * num_topk;
}

__forceinline__ __device__ __host__ int64_t checked_add(const int64_t lhs, const int64_t rhs) {
    EP_UNIFIED_ASSERT(lhs >= 0 and rhs >= 0);
    EP_UNIFIED_ASSERT(lhs <= std::numeric_limits<int64_t>::max() - rhs);
    return lhs + rhs;
}

__forceinline__ __device__ __host__ int64_t checked_mul(const int64_t lhs, const int64_t rhs) {
    EP_UNIFIED_ASSERT(lhs >= 0 and rhs >= 0);
    EP_UNIFIED_ASSERT(lhs == 0 or rhs <= std::numeric_limits<int64_t>::max() / lhs);
    return lhs * rhs;
}

__forceinline__ __device__ __host__ int64_t checked_align(const int64_t value,
                                                           const int64_t alignment) {
    EP_UNIFIED_ASSERT(value >= 0 and alignment > 0);
    const auto remainder = value % alignment;
    return remainder == 0 ? value : checked_add(value, alignment - remainder);
}

// The first array has a fixed extent because peers address it through LSA before
// the dynamic plan has been built. The remaining arrays use the invocation's
// actual geometry and are private, replicated plan outputs.
struct ArenaLayout {
    static constexpr int64_t kLocalCountBytes =
        static_cast<int64_t>(deep_ep::kNumMaxChannels) *
        kNumMaxDestinations * sizeof(int32_t);
    static constexpr int64_t kAlignment = ptx::kNumTMAAlignBytes;

    int hidden;
    int num_topk;
    int num_max_tokens_per_rank;
    int num_channels;
    int num_destinations;
    int num_rails;
    int num_max_tokens_per_channel;
    int proxy_capacity;
    void* base;

    int64_t local_count_offset;
    int64_t all_count_offset;
    int64_t quota_offset;
    int64_t rank_delta_offset;
    int64_t proxy_dispatch_offset;
    int64_t proxy_return_offset;
    int64_t dispatch_token_bytes;
    int64_t combine_token_bytes;
    int64_t raw_bytes;

    __forceinline__ __device__ __host__
    ArenaLayout(const int hidden,
                const int num_topk,
                const int num_max_tokens_per_rank,
                const int num_channels,
                const int num_destinations,
                const int num_rails,
                void* base = nullptr):
        hidden(hidden),
        num_topk(num_topk),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_channels(num_channels),
        num_destinations(num_destinations),
        num_rails(num_rails),
        num_max_tokens_per_channel(math::ceil_div(num_max_tokens_per_rank, num_channels)),
        proxy_capacity((num_destinations - 1) *
                       (num_max_tokens_per_rank + deep_ep::kNumMaxChannels)),
        base(base) {
        EP_UNIFIED_ASSERT(hidden > 0);
        EP_UNIFIED_ASSERT(hidden <= std::numeric_limits<int>::max() /
                                   static_cast<int>(sizeof(nv_bfloat16)));
        EP_UNIFIED_ASSERT(num_topk >= 1 and num_topk <= 32);
        EP_UNIFIED_ASSERT(num_max_tokens_per_rank > 0);
        EP_UNIFIED_ASSERT(num_channels >= 1 and num_channels <= deep_ep::kNumMaxChannels);
        EP_UNIFIED_ASSERT(num_destinations >= 2 and num_destinations <= kNumMaxDestinations);
        EP_UNIFIED_ASSERT(num_rails >= 1 and num_rails <= 32);

        const auto dispatch_layout = layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, true);
        const auto combine_layout = layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, false);
        dispatch_token_bytes = dispatch_layout.get_num_bytes<false, int64_t>();
        combine_token_bytes = combine_layout.get_num_bytes<false, int64_t>();

        const auto plan_elems = checked_mul(
            checked_mul(num_channels, num_destinations), num_rails);
        const auto plan_bytes = checked_mul(plan_elems, sizeof(int32_t));
        const auto rank_delta_bytes = checked_mul(
            checked_mul(num_destinations, num_rails), sizeof(int32_t));

        local_count_offset = 0;
        all_count_offset = checked_align(kLocalCountBytes, kAlignment);
        quota_offset = checked_align(checked_add(all_count_offset, plan_bytes), kAlignment);
        rank_delta_offset = checked_align(checked_add(quota_offset, plan_bytes), kAlignment);
        proxy_dispatch_offset = checked_align(
            checked_add(rank_delta_offset, rank_delta_bytes), kAlignment);
        proxy_return_offset = checked_align(
            checked_add(proxy_dispatch_offset,
                        checked_mul(proxy_capacity, dispatch_token_bytes)),
            kAlignment);
        raw_bytes = checked_add(proxy_return_offset,
                                checked_mul(proxy_capacity, combine_token_bytes));
    }

    __forceinline__ __device__ __host__ int32_t* get_local_count_ptr() const {
        return math::advance_ptr<int32_t>(base, local_count_offset);
    }

    __forceinline__ __device__ __host__ int32_t* get_all_count_ptr() const {
        return math::advance_ptr<int32_t>(base, all_count_offset);
    }

    __forceinline__ __device__ __host__ int32_t* get_quota_ptr() const {
        return math::advance_ptr<int32_t>(base, quota_offset);
    }

    __forceinline__ __device__ __host__ int32_t* get_rank_delta_ptr() const {
        return math::advance_ptr<int32_t>(base, rank_delta_offset);
    }

    __forceinline__ __device__ __host__ int get_plan_offset(
            const int channel, const int destination, const int rail) const {
        return (channel * num_destinations + destination) * num_rails + rail;
    }

    __forceinline__ __device__ __host__ int get_proxy_slot(
            const int local_destination,
            const int destination,
            const int channel,
            const int ordinal) const {
        EP_UNIFIED_ASSERT(destination != local_destination);
        EP_UNIFIED_ASSERT(channel >= 0 and channel < num_channels);
        EP_UNIFIED_ASSERT(ordinal >= 0 and ordinal < num_max_tokens_per_channel);
        const int remote_destination = destination < local_destination ? destination : destination - 1;
        return (remote_destination * num_channels + channel) *
                   num_max_tokens_per_channel + ordinal;
    }

    __forceinline__ __device__ __host__ layout::TokenLayout get_proxy_dispatch_layout(
            const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, true,
            math::advance_ptr(base, checked_add(
                proxy_dispatch_offset, checked_mul(proxy_slot, dispatch_token_bytes))));
    }

    __forceinline__ __device__ __host__ layout::TokenLayout get_proxy_return_layout(
            const int proxy_slot) const {
        EP_UNIFIED_ASSERT(proxy_slot >= 0 and proxy_slot < proxy_capacity);
        return layout::TokenLayout(
            hidden * sizeof(nv_bfloat16), 0, num_topk, false,
            math::advance_ptr(base, checked_add(
                proxy_return_offset, checked_mul(proxy_slot, combine_token_bytes))));
    }
};

}  // namespace deep_ep::elastic::rail_balance
