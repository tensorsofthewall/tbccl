#pragma once

// Phase 41 Part Q: the public asynchronous Work handle. The audit
// (docs/framework_integration_architecture.md Section 6) found
// TransferWork's existing shape (wait()/is_completed()/has_error()/
// error(), shared completion state, repeated-wait-safe,
// friend-gated construction) already public-API-grade -- it is
// promoted directly, by alias, rather than rebuilt. This is also used
// as the Work type returned by Communicator::all_reduce(), via
// detail::TransferWorkAccess::make() (async_transfer.hpp).
//
// Destruction semantics (Part Q item 96): operation state survives Work
// wrapper destruction. TransferWork::state_ is a shared_ptr, so
// destroying a Work handle whose operation is still in flight does not
// cancel it -- the operation completes against the same shared state;
// the destroyed handle just can no longer observe it. The Communicator
// (for P2P: TensorCommWorker; for collectives: the collective executor,
// communicator.hpp) owns the outstanding-operation's actual progress,
// not the Work object.

#include <tbccl/async_transfer.hpp>

namespace tbccl
{

using Work = TransferWork;

} // namespace tbccl
