#include <tbccl/world.hpp>

#include "ring_executor.hpp"

namespace tbccl
{

    // Defined here, not inline in world.hpp, because ring_executor_ is
    // a std::unique_ptr<detail::RingExecutor> and RingExecutor is only
    // forward-declared in the header — both the default constructor
    // (which default-constructs ring_executor_ to null) and the
    // destructor (which would destroy a non-null one, via
    // RingExecutor's own destructor) need RingExecutor's complete
    // definition, which is only available here.
    World::World() = default;

    World::~World() = default;

} // namespace tbccl
