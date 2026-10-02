#pragma once

#include <deep_ep/common/comm.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/math.cuh>
#include <deep_ep/common/ptx.cuh>
#include <deep_ep/common/rail_balance.cuh>

namespace deep_ep::elastic {

// One warp owns one source channel. Each lane counts one destination, so the
// output is overwritten in full and needs no memset between dispatches.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_count_impl(const topk_idx_t* topk_idx,
                             int* local_count,
                             int* rank_delta,
                             const int num_tokens,
                             const int num_topk,
                             const int num_channels,
                             const int num_experts,
                             const int num_destinations,
                             const int num_rails) {
    const int channel = static_cast<int>(blockIdx.x);
    const int lane = ptx::get_lane_idx();
    if (channel >= num_channels)
        return;

    for (int rank = channel * 32 + lane;
         rank < num_destinations * num_rails;
         rank += num_channels * 32)
        rank_delta[rank] = 0;

    const int experts_per_destination = num_experts / num_destinations;
    int count = 0;
    for (int token = channel; token < num_tokens; token += num_channels) {
        const int expert = lane < num_topk ?
            static_cast<int>(__ldg(topk_idx + token * num_topk + lane)) : -1;
        const int destination = expert >= 0 ? expert / experts_per_destination : -1;
        const unsigned destination_mask = ptx::reduce_or(
            destination >= 0 and destination < num_destinations ?
                (1u << destination) : 0u);
        count += lane < num_destinations and ((destination_mask >> lane) & 1u);
    }
    if (lane < num_destinations)
        local_count[channel * num_destinations + lane] = count;
}

// All ranks execute the same deterministic planner after a local barrier. It
// first snapshots peer counts through LSA, then balances every channel while
// rotating remainder ownership to avoid a persistent low-rank bias.
template <int kInstantiation = 0>
__global__ __launch_bounds__(32, 1)
void rail_balance_plan_impl(const ncclDevComm_t nccl_dev_comm,
                            const ncclWindow_t nccl_window,
                            const int* local_count,
                            int* all_count,
                            int* quota,
                            const int* incast_rail_masks,
                            const int num_channels,
                            const int num_destinations,
                            const int num_rails,
                            const int local_destination,
                            const int policy) {
    const int destination = static_cast<int>(blockIdx.x);
    const int lane = ptx::get_lane_idx();
    if (destination >= num_destinations)
        return;

    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    unsigned incast_rail_control = 0;
    if (lane == 0 and incast_rail_masks != nullptr)
        incast_rail_control = static_cast<unsigned>(__ldg(incast_rail_masks + destination));
    incast_rail_control = ptx::exchange(incast_rail_control, 0);
    const unsigned valid_rail_mask = num_rails == 32 ? 0xffffffffu : ((1u << num_rails) - 1u);
    const unsigned incast_rail_mask = incast_rail_control & valid_rail_mask;
    // Values above an ordinary 8-bit mask encode eight four-bit Rail weights.
    const bool use_weighted_quotas =
        num_rails == 8 and incast_rail_control > valid_rail_mask;

    int owner_total = 0;
    if (lane < num_rails) {
        const auto peer_count = gin.get_sym_ptr<ncclTeamTagLsa>(local_count, lane);
        for (int channel = 0; channel < num_channels; ++ channel)
            owner_total += __ldg(peer_count + channel * num_destinations + destination);
    }
    int static_weight = 0;
    if (lane < num_rails and use_weighted_quotas)
        static_weight = static_cast<int>(
            (incast_rail_control >> (4 * lane)) & 0xFu);
    const int static_weight_sum = ptx::reduce_add(static_weight);
    const bool use_incast_mask =
        policy == static_cast<int>(rail_balance::Policy::Incast) and
        destination != local_destination and
        (use_weighted_quotas ? static_weight_sum > 0 : incast_rail_mask != 0);
    const bool static_selected = lane < num_rails and (
        use_incast_mask ? (
            use_weighted_quotas ? static_weight > 0 :
            ((incast_rail_mask >> lane) & 1u)) :
        (policy == static_cast<int>(rail_balance::Policy::All) or owner_total > 0));
    const unsigned static_selected_mask = ptx::gather(static_selected);

    for (int channel = 0; channel < num_channels; ++ channel) {
        const unsigned selected_mask = static_selected_mask;
        const bool selected = static_selected;
        const int num_selected = __popc(selected_mask);

        int count = 0;
        if (lane < num_rails) {
            const auto peer_count = gin.get_sym_ptr<ncclTeamTagLsa>(local_count, lane);
            count = __ldg(peer_count + channel * num_destinations + destination);
            all_count[(channel * num_destinations + destination) * num_rails + lane] = count;
        }
        __syncwarp();

        int target = count;
        if (destination != local_destination and num_selected > 0) {
            const int total = ptx::reduce_add(count);
            const unsigned lower_lanes = lane == 0 ? 0u : (0xffffffffu >> (32 - lane));
            const int selected_position = __popc(selected_mask & lower_lanes);
            const int rotation = (channel + destination) % num_selected;
            const int rotated_position =
                (selected_position - rotation + num_selected) % num_selected;
            if (use_weighted_quotas) {
                target = total * static_weight / static_weight_sum;
                const int assigned = ptx::reduce_add(target);
                const int remainder = total - assigned;
                target += static_cast<int>(
                    selected and rotated_position < remainder);
            } else {
                const int base = total / num_selected;
                const int remainder = total - base * num_selected;
                target = selected ?
                    base + static_cast<int>(rotated_position < remainder) : 0;
            }
        }
        if (lane < num_rails)
            quota[(channel * num_destinations + destination) * num_rails + lane] = target;
        __syncwarp();
    }
}

template <int kNumHiddenBytes, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_source_shuffle_impl(
        const ncclDevComm_t nccl_dev_comm,
        const ncclWindow_t nccl_window,
        const nv_bfloat16* x,
        const topk_idx_t* topk_idx,
        const float* topk_weights,
        void* arena,
        const int* all_count,
        const int* quota,
        int* rank_delta,
        const int num_tokens,
        const int num_experts,
        const int num_max_tokens_per_rank,
        const int num_channels,
        const int num_destinations,
        const int num_rails,
        const int local_destination,
        const int owner) {
    const int channel = static_cast<int>(blockIdx.x);
    const int lane = ptx::get_lane_idx();
    if (channel >= num_channels)
        return;

    const auto arena_layout = rail_balance::ArenaLayout(
        kNumHiddenBytes / static_cast<int>(sizeof(nv_bfloat16)), kNumTopk,
        num_max_tokens_per_rank, num_channels, num_destinations, num_rails, arena);
    const auto token_layout = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, true);
    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    auto tma_buffer = layout::TokenLayout(kNumHiddenBytes, 0, kNumTopk, true, smem);
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    const int experts_per_destination = num_experts / num_destinations;
    const int experts_per_rank = num_experts / (num_destinations * num_rails);
    int next_ordinal = 0;

    for (int token = channel; token < num_tokens; token += num_channels) {
        const int expert = lane < kNumTopk ?
            static_cast<int>(__ldg(topk_idx + token * kNumTopk + lane)) : -1;
        const int destination = expert >= 0 ? expert / experts_per_destination : -1;
        unsigned destination_mask = ptx::reduce_or(
            destination >= 0 and destination < num_destinations ?
                (1u << destination) : 0u);
        destination_mask &= ~(1u << local_destination);
        unsigned remaining = destination_mask;
        bool staged = false;
        while (remaining != 0) {
            const int dst = ptx::ffs(remaining);
            remaining ^= 1u << dst;
            const int owner_ordinal = ptx::exchange(next_ordinal, dst);
            int egress = -1, egress_ordinal = -1;
            if (lane == 0) {
                const int owner_offset =
                    (channel * num_destinations + dst) * num_rails + owner;
                const int owner_quota = __ldg(quota + owner_offset);
                if (owner_ordinal >= owner_quota) {
                    int surplus_ordinal = owner_ordinal - owner_quota;
                    for (int peer = 0; peer < owner; ++ peer) {
                        const int offset =
                            (channel * num_destinations + dst) * num_rails + peer;
                        surplus_ordinal += max(
                            __ldg(all_count + offset) - __ldg(quota + offset), 0);
                    }
                    for (int peer = 0; peer < num_rails; ++ peer) {
                        const int offset =
                            (channel * num_destinations + dst) * num_rails + peer;
                        const int peer_count = __ldg(all_count + offset);
                        const int deficit = max(
                            __ldg(quota + offset) - peer_count, 0);
                        if (surplus_ordinal < deficit) {
                            egress = peer;
                            egress_ordinal = peer_count + surplus_ordinal;
                            break;
                        }
                        surplus_ordinal -= deficit;
                    }
                }
            }
            egress = ptx::exchange(egress, 0);
            egress_ordinal = ptx::exchange(egress_ordinal, 0);
            if (egress >= 0) {
                if (not staged) {
                    if (ptx::elect_one_sync())
                        ptx::tma_load_1d(
                            tma_buffer.get_hidden_ptr(),
                            x + static_cast<int64_t>(token) *
                                (kNumHiddenBytes / sizeof(nv_bfloat16)),
                            mbarrier_ptr, kNumHiddenBytes);
                    if (lane < kNumTopk) {
                        tma_buffer.get_topk_idx_ptr()[lane] = expert;
                        tma_buffer.get_topk_weights_ptr()[lane] =
                            topk_weights == nullptr ? 0.0f :
                            __ldg(topk_weights + token * kNumTopk + lane);
                    }
                    if (ptx::elect_one_sync())
                        *tma_buffer.get_src_token_global_idx_ptr() =
                            (local_destination * num_rails + owner) *
                                num_max_tokens_per_rank + token;
                    ptx::tma_store_fence();
                    __syncwarp();
                    if (ptx::elect_one_sync()) {
                        ptx::mbarrier_arrive_and_set_tx(
                            mbarrier_ptr, kNumHiddenBytes);
                        ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                    }
                    __syncwarp();
                    staged = true;
                }
                const int proxy_slot = arena_layout.get_proxy_slot(
                    local_destination, dst, channel, egress_ordinal);
                const int dst_rank = expert >= 0 ? expert / experts_per_rank : -1;
                const bool unique_rank = ptx::deduplicate(dst_rank, lane);
                if (unique_rank and dst_rank >= 0 and dst_rank / num_rails == dst) {
                    ptx::red_add_rel_sys(rank_delta + dst_rank, -1);
                    const auto peer_rank_delta =
                        gin.get_sym_ptr<ncclTeamTagLsa>(rank_delta, egress);
                    ptx::red_add_rel_sys(peer_rank_delta + dst_rank, 1);
                }
                if (ptx::elect_one_sync())
                    *tma_buffer.get_linked_list_idx_ptr() = proxy_slot;
                ptx::tma_store_fence();
                __syncwarp();
                void* peer_arena =
                    gin.get_sym_ptr<ncclTeamTagLsa>(arena, egress);
                const auto peer_arena_layout = rail_balance::ArenaLayout(
                    kNumHiddenBytes / static_cast<int>(sizeof(nv_bfloat16)), kNumTopk,
                    num_max_tokens_per_rank, num_channels, num_destinations,
                    num_rails, peer_arena);
                const auto proxy =
                    peer_arena_layout.get_proxy_dispatch_layout(proxy_slot);
                if (ptx::elect_one_sync()) {
                    ptx::tma_store_1d(proxy.get_base_ptr(), tma_buffer.get_base_ptr(),
                                      token_layout.get_num_bytes<false>());
                }
                ptx::tma_store_commit();
                // The next token only needs the shared stage to be free. The
                // peer write itself can finish in the background.
                ptx::tma_store_wait_read();
                __syncwarp();
            }
        }
        next_ordinal += lane < num_destinations and
                        ((destination_mask >> lane) & 1u);
    }
    ptx::tma_store_wait();
}

template <int kHidden, int kNumTopk>
__global__ __launch_bounds__(32, 1)
void rail_balance_return_unshuffle_impl(
        const ncclDevComm_t nccl_dev_comm,
        const ncclWindow_t nccl_window,
        void* arena,
        void* legacy_reduce_buffer,
        const int* all_count,
        const int* quota,
        const int num_experts,
        const int num_max_tokens_per_rank,
        const int num_channels,
        const int num_destinations,
        const int num_rails,
        const int local_destination,
        const int egress) {
    constexpr int kNumHiddenBytes = kHidden * sizeof(nv_bfloat16);
    const int channel = static_cast<int>(blockIdx.x);
    const int lane = ptx::get_lane_idx();
    if (channel >= num_channels)
        return;

    const auto arena_layout = rail_balance::ArenaLayout(
        kHidden, kNumTopk, num_max_tokens_per_rank,
        num_channels, num_destinations, num_rails, arena);
    const auto token_layout = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false);
    const int num_layout_ranks = min(num_destinations, kNumTopk);
    auto reduce_layout = layout::BufferLayout<false>(
        token_layout, num_layout_ranks, num_max_tokens_per_rank,
        legacy_reduce_buffer);

    extern __shared__ __align__(ptx::kNumTMAAlignBytes) int8_t smem[];
    auto tma_buffer = layout::TokenLayout(
        kNumHiddenBytes, 0, kNumTopk, false, smem);
    const auto mbarrier_ptr = tma_buffer.get_mbarrier_ptr();
    ptx::arrival_phase phase = 0;
    if (ptx::elect_one_sync())
        ptx::mbarrier_init_with_fence(mbarrier_ptr, 1);
    __syncwarp();

    const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
    const int experts_per_destination = num_experts / num_destinations;
    for (int destination = 0; destination < num_destinations; ++ destination) {
        if (destination == local_destination)
            continue;
        const int offset =
            (channel * num_destinations + destination) * num_rails + egress;
        const int begin = __ldg(all_count + offset);
        const int end = __ldg(quota + offset);
        for (int ordinal = begin; ordinal < end; ++ ordinal) {
            const int proxy_slot = arena_layout.get_proxy_slot(
                local_destination, destination, channel, ordinal);
            const auto proxy_dispatch =
                arena_layout.get_proxy_dispatch_layout(proxy_slot);
            const auto proxy_return =
                arena_layout.get_proxy_return_layout(proxy_slot);
            const int src_global_token =
                __ldg(proxy_dispatch.get_src_token_global_idx_ptr());
            const int owner =
                (src_global_token / num_max_tokens_per_rank) % num_rails;
            const int src_token = src_global_token % num_max_tokens_per_rank;

            int layout_rank = destination;
            if (num_destinations > kNumTopk) {
                const int expert = lane < kNumTopk ?
                    __ldg(proxy_dispatch.get_topk_idx_ptr() + lane) : -1;
                const unsigned matching = ptx::gather(
                    expert >= 0 and
                    expert / experts_per_destination == destination);
                layout_rank = ptx::get_master_lane_idx(matching);
            }
            layout_rank = ptx::exchange(layout_rank, 0);

            if (ptx::elect_one_sync()) {
                ptx::tma_load_1d(
                    tma_buffer.get_base_ptr(), proxy_return.get_base_ptr(),
                    mbarrier_ptr, token_layout.get_num_bytes<false>());
                ptx::mbarrier_arrive_and_set_tx(
                    mbarrier_ptr, token_layout.get_num_bytes<false>());
                ptx::mbarrier_wait_and_flip_phase(mbarrier_ptr, phase);
                auto target = reduce_layout.get_rank_buffer(layout_rank)
                                  .get_token_buffer(src_token);
                target.set_base_ptr(gin.get_sym_ptr<ncclTeamTagLsa>(
                    target.get_base_ptr(), owner));
                ptx::tma_store_1d(
                    target.get_base_ptr(), tma_buffer.get_base_ptr(),
                    token_layout.get_num_bytes<false>());
                ptx::tma_store_commit();
            }
            __syncwarp();
            ptx::tma_store_wait_read();
        }
    }
    ptx::tma_store_wait();
}

}  // namespace deep_ep::elastic
