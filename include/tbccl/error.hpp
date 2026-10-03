#pragma once

// The typed error carrier. Every EXPECTED failure that can reach a caller (a thrown call, or the terminal state of a Work) carries an ErrorCode
// that was decided where the failure happened. Message text keeps its historical "tag: detail" shape for human readers and for the torch-tbccl
// error mapper, but nothing in TBCCL (and no C ABI shim) derives the category from the text: the code and the text are independent.

#include <tbccl/types.hpp>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace tbccl
{

class Error : public std::runtime_error
{
public:
    Error(ErrorCode code, const std::string &message) : std::runtime_error(message), code_(code) {}
    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

// The one mapping from "something was thrown" to a code: tbccl::Error carries its own, an allocation failure is ResourceExhausted, anything else
// that was not expected to escape is InternalError.
inline ErrorCode error_code_of(const std::exception &e) noexcept
{
    if (const auto *typed = dynamic_cast<const Error *>(&e)) return typed->code();
    if (dynamic_cast<const std::bad_alloc *>(&e) != nullptr) return ErrorCode::ResourceExhausted;
    return ErrorCode::InternalError;
}

// For a failure that wraps a lower-level one with extra context: keep the cause's code when it is a typed Error (a Timeout stays a Timeout, a
// ProtocolMismatch stays a ProtocolMismatch), otherwise use `fallback`.
inline ErrorCode wrapped_code(const std::exception &cause, ErrorCode fallback) noexcept
{
    if (const auto *typed = dynamic_cast<const Error *>(&cause)) return typed->code();
    if (dynamic_cast<const std::bad_alloc *>(&cause) != nullptr) return ErrorCode::ResourceExhausted;
    return fallback;
}

} // namespace tbccl
