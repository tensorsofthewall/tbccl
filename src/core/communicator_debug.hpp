#pragma once

// Phase 51: test/diagnostic accessors into a Communicator's runtime, private to libtbccl (NOT installed, not public API).

#include <tbccl/communicator.hpp>

#include <vector>

namespace tbccl::detail
{

// Ranks this communicator currently has an established DATA connection to (world_size 2: its one peer from bootstrap; above that, only what lazy
// establishment has connected so far). The control plane is always a full mesh and is not reported here.
std::vector<std::size_t> debug_connected_data_peers(const Communicator &comm);

// Test hook: the next `count` P2P admissions on this communicator fail as if allocation failed (ErrorCode::ResourceExhausted).
void debug_fail_next_admissions(const Communicator &comm, int count);

} // namespace tbccl::detail
