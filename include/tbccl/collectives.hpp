#pragma once

#include <tbccl/world.hpp>

namespace tbccl
{

// Blocks the calling rank until every rank in `world` has called
// barrier(). All ranks must call it in matching collective order.
//
// World's connections are an unframed byte stream per peer, so
// barrier() requires exclusive collective use of every peer connection
// for the duration of the call: concurrent unrelated send()/recv()
// traffic over the same World from another thread, or application
// messages left in flight from a previous phase, is not supported and
// will corrupt the control protocol. Sequence application
// communication and barrier() calls strictly one after another (see
// tests/barrier_test.cpp for the intended before/barrier/after
// pattern).
void barrier(World &world);

} // namespace tbccl
