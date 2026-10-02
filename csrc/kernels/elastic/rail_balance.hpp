#pragma once

#include <nccl.h>
#include <nccl_device.h>

#include <deep_ep/common/exception.cuh>
#include <deep_ep/common/layout.cuh>
#include <deep_ep/common/rail_balance.cuh>

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

class RailBalanceCountRuntime final : public jit::LaunchRuntime<RailBalanceCountRuntime> {
public:
    struct Args {
        const topk_idx_t* topk_idx;
        int* local_count;
        int* rank_delta;
        int num_tokens, num_topk, num_channels, num_experts, num_destinations;
        int num_rails;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&rail_balance_count_impl<>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.topk_idx, args.local_count,
            args.rank_delta,
            args.num_tokens, args.num_topk, args.num_channels,
            args.num_experts, args.num_destinations, args.num_rails));
    }
};

class RailBalancePlanRuntime final : public jit::LaunchRuntime<RailBalancePlanRuntime> {
public:
    struct Args {
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        const int* local_count;
        int* all_count;
        int* quota;
        const int* incast_rail_masks;
        int num_channels, num_destinations, num_rails;
        int local_destination, policy;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args&) {
        return R"(
#include <deep_ep/impls/rail_balance.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&rail_balance_plan_impl<>);
}
)";
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.nccl_dev_comm, args.nccl_window,
            args.local_count, args.all_count, args.quota,
            args.incast_rail_masks,
            args.num_channels, args.num_destinations, args.num_rails,
            args.local_destination, args.policy));
    }
};

class RailBalanceShuffleRuntime final : public jit::LaunchRuntime<RailBalanceShuffleRuntime> {
public:
    struct Args {
        int num_hidden_bytes, num_topk;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        const nv_bfloat16* x;
        const topk_idx_t* topk_idx;
        const float* topk_weights;
        void* arena;
        const int* all_count;
        const int* quota;
        int* rank_delta;
        int num_tokens, num_experts, num_max_tokens_per_rank;
        int num_channels, num_destinations, num_rails;
        int local_destination, owner;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_source_shuffle_impl<{}, {}>);
}}
)", args.num_hidden_bytes, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.nccl_dev_comm, args.nccl_window,
            args.x, args.topk_idx, args.topk_weights,
            args.arena, args.all_count, args.quota, args.rank_delta,
            args.num_tokens, args.num_experts, args.num_max_tokens_per_rank,
            args.num_channels, args.num_destinations, args.num_rails,
            args.local_destination, args.owner));
    }
};

class RailBalanceUnshuffleRuntime final : public jit::LaunchRuntime<RailBalanceUnshuffleRuntime> {
public:
    struct Args {
        int hidden, num_topk;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* arena;
        void* legacy_reduce_buffer;
        const int* all_count;
        const int* quota;
        int num_experts, num_max_tokens_per_rank;
        int num_channels, num_destinations, num_rails;
        int local_destination, egress;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/rail_balance.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &rail_balance_return_unshuffle_impl<{}, {}>);
}}
)", args.hidden, args.num_topk);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.nccl_dev_comm, args.nccl_window,
            args.arena, args.legacy_reduce_buffer,
            args.all_count, args.quota,
            args.num_experts, args.num_max_tokens_per_rank,
            args.num_channels, args.num_destinations, args.num_rails,
            args.local_destination, args.egress));
    }
};

class RailBalanceLocalBarrierRuntime final :
    public jit::LaunchRuntime<RailBalanceLocalBarrierRuntime> {
public:
    struct Args {
        int num_rails;
        int64_t num_timeout_cycles;
        jit::NoRefPtr nccl_dev_comm;
        ncclWindow_t nccl_window;
        void* workspace;
        int rail;
        jit::LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/barrier.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(
        &barrier_impl<true, 1, 512, 1, {}, {}, true>);
}}
)", args.num_rails, args.num_timeout_cycles);
    }

    static void launch_impl(const jit::KernelHandle& kernel,
                            const jit::LaunchConfigHandle& config,
                            Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
            kernel, config, args.nccl_dev_comm, args.nccl_window,
            args.workspace, 0, args.rail));
    }
};

static void launch_rail_balance_local_barrier(
        const jit::NoRefPtr& nccl_dev_comm,
        const ncclWindow_t& nccl_window,
        void* workspace,
        const int& num_rails,
        const int& rail,
        const int64_t& num_timeout_cycles,
        const at::cuda::CUDAStream& stream) {
    const RailBalanceLocalBarrierRuntime::Args args = {
        .num_rails = num_rails,
        .num_timeout_cycles = num_timeout_cycles,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .workspace = workspace,
        .rail = rail,
        .launch_args = jit::LaunchArgs(1, 512, 0, 1, true),
    };
    const auto runtime = jit::compiler->build(
        "rail_balance_local_barrier", RailBalanceLocalBarrierRuntime::generate(args));
    RailBalanceLocalBarrierRuntime::launch(runtime, args, stream);
}

static void launch_rail_balance_prepare(
        const jit::NoRefPtr& nccl_dev_comm,
        const ncclWindow_t& nccl_window,
        void* workspace,
        const nv_bfloat16* x,
        const topk_idx_t* topk_idx,
        const float* topk_weights,
        const rail_balance::ArenaLayout& arena,
        const int* incast_rail_masks,
        const int& num_tokens,
        const int& num_experts,
        const int& local_destination,
        const int& owner,
        const int& policy,
        const int64_t& num_timeout_cycles,
        const at::cuda::CUDAStream& stream) {
    auto count_args = RailBalanceCountRuntime::Args {
        .topk_idx = topk_idx,
        .local_count = arena.get_local_count_ptr(),
        .rank_delta = arena.get_rank_delta_ptr(),
        .num_tokens = num_tokens,
        .num_topk = arena.num_topk,
        .num_channels = arena.num_channels,
        .num_experts = num_experts,
        .num_destinations = arena.num_destinations,
        .num_rails = arena.num_rails,
        .launch_args = jit::LaunchArgs(arena.num_channels, 32, 0),
    };
    const auto count_runtime = jit::compiler->build(
        "rail_balance_count", RailBalanceCountRuntime::generate(count_args));
    RailBalanceCountRuntime::launch(count_runtime, count_args, stream);

    launch_rail_balance_local_barrier(
        nccl_dev_comm, nccl_window, workspace,
        arena.num_rails, owner, num_timeout_cycles, stream);

    auto plan_args = RailBalancePlanRuntime::Args {
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .local_count = arena.get_local_count_ptr(),
        .all_count = arena.get_all_count_ptr(),
        .quota = arena.get_quota_ptr(),
        .incast_rail_masks = incast_rail_masks,
        .num_channels = arena.num_channels,
        .num_destinations = arena.num_destinations,
        .num_rails = arena.num_rails,
        .local_destination = local_destination,
        .policy = policy,
        .launch_args = jit::LaunchArgs(arena.num_destinations, 32, 0),
    };
    const auto plan_runtime = jit::compiler->build(
        "rail_balance_plan", RailBalancePlanRuntime::generate(plan_args));
    RailBalancePlanRuntime::launch(plan_runtime, plan_args, stream);

    const auto token_layout = layout::TokenLayout(
        arena.hidden * sizeof(nv_bfloat16), 0, arena.num_topk, true);
    auto shuffle_args = RailBalanceShuffleRuntime::Args {
        .num_hidden_bytes = arena.hidden * static_cast<int>(sizeof(nv_bfloat16)),
        .num_topk = arena.num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .x = x,
        .topk_idx = topk_idx,
        .topk_weights = topk_weights,
        .arena = arena.base,
        .all_count = arena.get_all_count_ptr(),
        .quota = arena.get_quota_ptr(),
        .rank_delta = arena.get_rank_delta_ptr(),
        .num_tokens = num_tokens,
        .num_experts = num_experts,
        .num_max_tokens_per_rank = arena.num_max_tokens_per_rank,
        .num_channels = arena.num_channels,
        .num_destinations = arena.num_destinations,
        .num_rails = arena.num_rails,
        .local_destination = local_destination,
        .owner = owner,
        .launch_args = jit::LaunchArgs(
            arena.num_channels, 32, token_layout.get_num_bytes<true>()),
    };
    const auto shuffle_runtime = jit::compiler->build(
        "rail_balance_source_shuffle_v2",
        RailBalanceShuffleRuntime::generate(shuffle_args));
    RailBalanceShuffleRuntime::launch(shuffle_runtime, shuffle_args, stream);
    // The following hybrid dispatch starts with a scale-up barrier. The stream
    // boundary completes these TMA stores, so that barrier also closes shuffle.
}

static void launch_rail_balance_return_unshuffle(
        const jit::NoRefPtr& nccl_dev_comm,
        const ncclWindow_t& nccl_window,
        void* workspace,
        const rail_balance::ArenaLayout& arena,
        void* legacy_reduce_buffer,
        const int& num_experts,
        const int& local_destination,
        const int& egress,
        const int64_t& num_timeout_cycles,
        const at::cuda::CUDAStream& stream) {
    const auto token_layout = layout::TokenLayout(
        arena.hidden * sizeof(nv_bfloat16), 0, arena.num_topk, false);
    auto args = RailBalanceUnshuffleRuntime::Args {
        .hidden = arena.hidden,
        .num_topk = arena.num_topk,
        .nccl_dev_comm = nccl_dev_comm,
        .nccl_window = nccl_window,
        .arena = arena.base,
        .legacy_reduce_buffer = legacy_reduce_buffer,
        .all_count = arena.get_all_count_ptr(),
        .quota = arena.get_quota_ptr(),
        .num_experts = num_experts,
        .num_max_tokens_per_rank = arena.num_max_tokens_per_rank,
        .num_channels = arena.num_channels,
        .num_destinations = arena.num_destinations,
        .num_rails = arena.num_rails,
        .local_destination = local_destination,
        .egress = egress,
        .launch_args = jit::LaunchArgs(
            arena.num_channels, 32, token_layout.get_num_bytes<true>()),
    };
    const auto runtime = jit::compiler->build(
        "rail_balance_return_unshuffle_v2",
        RailBalanceUnshuffleRuntime::generate(args));
    RailBalanceUnshuffleRuntime::launch(runtime, args, stream);
    launch_rail_balance_local_barrier(
        nccl_dev_comm, nccl_window, workspace,
        arena.num_rails, egress, num_timeout_cycles, stream);
}

}  // namespace deep_ep::elastic
