// Structured errors. The ErrorCode of a thrown call or of a terminal Work is decided where the failure happens and carried separately from the message
// text; nothing derives the category from the text. Every category is injected and checked with a text that would mislead a string parser.

#include "mesh_test_support.hpp"

#include "communicator_debug.hpp"

#include <tbccl/error.hpp>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::ErrorCode;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{
    BufferView view(std::vector<std::uint8_t> &v) { return BufferView{MemoryKind::Host, v.data(), v.size(), 0}; }
    BufferView fview(std::vector<float> &v) { return BufferView{MemoryKind::Host, v.data(), v.size() * 4, 0}; }

    // Runs `f` and returns the code of the tbccl::Error it throws; fails the test if it throws anything else or nothing.
    template <typename F> ErrorCode thrown_code(F &&f, const std::string &what)
    {
        try
        {
            f();
        }
        catch (const tbccl::Error &e)
        {
            return e.code();
        }
        catch (const std::exception &e)
        {
            expect(false, what + ": threw a non-typed exception: " + e.what());
        }
        expect(false, what + ": did not throw");
        return ErrorCode::Success;
    }

    void test_carrier()
    {
        const tbccl::Error plain(ErrorCode::Timeout, "invalid_argument: this text says something else");
        expect(plain.code() == ErrorCode::Timeout, "the code is the constructor's, whatever the text says");
        expect(std::string(plain.what()).find("invalid_argument") == 0, "the text is preserved verbatim");
        expect(tbccl::error_code_of(plain) == ErrorCode::Timeout, "error_code_of(Error)");
        expect(tbccl::error_code_of(std::bad_alloc()) == ErrorCode::ResourceExhausted, "bad_alloc -> ResourceExhausted");
        expect(tbccl::error_code_of(std::runtime_error("aborted: looks like an abort")) == ErrorCode::InternalError, "an untyped std::exception is InternalError, its text is not parsed");
        expect(tbccl::error_code_of(std::logic_error("timeout: nope")) == ErrorCode::InternalError, "an untyped logic_error is InternalError");
        const tbccl::Error wrapped(tbccl::wrapped_code(plain, ErrorCode::TransportError), "transport_error: wrapping");
        expect(wrapped.code() == ErrorCode::Timeout, "a wrapped typed cause keeps its code");
        expect(tbccl::wrapped_code(std::runtime_error("x"), ErrorCode::TransportError) == ErrorCode::TransportError, "an untyped cause uses the fallback");
        std::cout << "[PASS] Error carrier: code independent of text; bad_alloc and untyped exceptions mapped centrally\n";
    }

    void test_work_state_independent_of_text()
    {
        const ErrorCode all[] = {ErrorCode::InvalidArgument, ErrorCode::Unsupported, ErrorCode::ResourceExhausted, ErrorCode::Aborted, ErrorCode::Timeout,
                                 ErrorCode::ProtocolMismatch, ErrorCode::TransportError, ErrorCode::PeerFailure, ErrorCode::DeviceError, ErrorCode::InternalError};
        for (ErrorCode code : all)
        {
            tbccl::Work w = tbccl::detail::TransferWorkAccess::make();
            expect(!w.is_completed() && w.error_code() == ErrorCode::Success, "a live Work reports Success until terminal");
            expect(!w.wait_for(std::chrono::milliseconds(1)), "wait_for on a live Work times out and does not complete it");
            expect(!w.is_completed(), "a timed-out wait_for does not change the Work");
            tbccl::detail::TransferWorkAccess::complete_error(tbccl::detail::TransferWorkAccess::state_of(w), code, "success: every text here claims something else");
            expect(w.wait_for(std::chrono::milliseconds(1)), "wait_for returns true on a terminal Work");
            expect(w.has_error() && w.error_code() == code, "error_code() is exactly what the failure carried: " + tbccl::error_code_name(code));
            tbccl::detail::TransferWorkAccess::complete_error(tbccl::detail::TransferWorkAccess::state_of(w), ErrorCode::InternalError, "second completion");
            expect(w.error_code() == code, "exactly one terminal transition");
        }
        tbccl::Work ok = tbccl::detail::TransferWorkAccess::make();
        tbccl::detail::TransferWorkAccess::complete_ok(tbccl::detail::TransferWorkAccess::state_of(ok));
        expect(!ok.has_error() && ok.error_code() == ErrorCode::Success, "a successful Work reports Success");
        std::cout << "[PASS] Work terminal state carries the code separately from the text (every category, one terminal transition)\n";
    }

    void test_call_errors()
    {
        run_world(3, [](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> buf(64, 1);
            expect(thrown_code([&] { comm.send(view(buf), buf.size(), DataType::UInt8, rank); }, "send to self") == ErrorCode::InvalidArgument, "send to self -> InvalidArgument");
            expect(thrown_code([&] { comm.recv(view(buf), buf.size(), DataType::UInt8, 99); }, "recv from rank 99") == ErrorCode::InvalidArgument, "peer out of range -> InvalidArgument");
            expect(thrown_code([&] { comm.send(BufferView{MemoryKind::Host, nullptr, 64, 0}, 64, DataType::UInt8, (rank + 1) % 3); }, "null buffer") == ErrorCode::InvalidArgument, "null data -> InvalidArgument");
            expect(thrown_code([&] { comm.send(BufferView{MemoryKind::Cuda, buf.data(), buf.size(), 0}, buf.size(), DataType::UInt8, (rank + 1) % 3); }, "unregistered kind") == ErrorCode::Unsupported, "unregistered memory kind -> Unsupported");
            std::vector<float> f(16, 1.0f);
            // FP16 reductions are rejected at N>2 (the N-rank runtime work) before anything is submitted: the call throws, the
            // communicator stays usable.
            expect(thrown_code([&] { comm.all_reduce(fview(f), fview(f), 32, DataType::Float16, ReduceOp::Sum); }, "FP16 all_reduce at N>2") == ErrorCode::Unsupported, "FP16 all_reduce at N>2 -> Unsupported");
            expect(!comm.failed(), "an Unsupported call does not poison the communicator");

            // Allocation failure at admission: reported at once, nothing accepted, communicator still usable.
            tbccl::detail::debug_fail_next_admissions(comm, 2);
            expect(thrown_code([&] { comm.send(view(buf), buf.size(), DataType::UInt8, (rank + 1) % 3); }, "injected allocation failure") == ErrorCode::ResourceExhausted, "bad_alloc at admission -> ResourceExhausted");
            expect(thrown_code([&] { comm.recv(view(buf), buf.size(), DataType::UInt8, (rank + 2) % 3); }, "injected allocation failure (recv)") == ErrorCode::ResourceExhausted, "bad_alloc at recv admission -> ResourceExhausted");
            expect(!comm.failed(), "ResourceExhausted at admission does not poison the communicator");
            comm.barrier().wait();
        });
        std::cout << "[PASS] thrown calls: InvalidArgument, Unsupported, ResourceExhausted; an Unsupported Work leaves the communicator usable\n";
    }

    void test_mismatch_and_abort()
    {
        run_world(3, [](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<float> f(200, 1.0f);
            const std::size_t count = rank == 0 ? 100 : 101; // rank 0 disagrees
            auto w = comm.all_reduce(fview(f), fview(f), count, DataType::Float32, ReduceOp::Sum);
            w.wait();
            expect(w.has_error(), "mismatched all_reduce fails on every rank");
            const ErrorCode c = w.error_code();
            if (rank == 0) expect(c == ErrorCode::ProtocolMismatch, "the judging rank reports ProtocolMismatch");
            else expect(c == ErrorCode::ProtocolMismatch || c == ErrorCode::Aborted, "every rank reports ProtocolMismatch or (if the abort arrived first) Aborted");
            expect(comm.failed(), "a mismatch poisons the communicator");
            // after the abort: new calls throw Aborted, and the abort reason is queryable
            expect(thrown_code([&] { std::vector<std::uint8_t> b(8); comm.send(view(b), 8, DataType::UInt8, (rank + 1) % 3); }, "send after abort") == ErrorCode::Aborted, "a call on an aborted communicator throws Aborted");
        });
        Latch posted(3);
        run_world(3, [&posted](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> buf(1 << 20, 0);
            auto pending = comm.recv(view(buf), buf.size(), DataType::UInt8, (rank + 1) % 3); // nobody sends: stays pending
            posted.arrive_and_wait(); // every rank has posted before anyone aborts
            comm.abort("test abort");
            pending.wait();
            expect(pending.has_error() && pending.error_code() == ErrorCode::Aborted, "a pending recv fails Aborted when the communicator is aborted");
            expect(comm.aborted(), "aborted()");
        });
        std::cout << "[PASS] collective mismatch -> ProtocolMismatch; abort -> pending Works and new calls report Aborted\n";
    }

    void test_bootstrap_timeout()
    {
        tbccl::CommunicatorOptions o;
        o.rank = 1;
        o.peers = {{"127.0.0.1", 1}, {"127.0.0.1", 0}}; // rank 0's endpoint: nobody listens on port 1
        o.bootstrap_timeout = std::chrono::milliseconds(300);
        const ErrorCode c = thrown_code([&] { (void)tbccl::Communicator::create(o); }, "bootstrap against a dead endpoint");
        expect(c == ErrorCode::Timeout, "an unreachable peer at bootstrap -> Timeout");
        std::cout << "[PASS] bootstrap timeout -> Timeout\n";
    }
} // namespace

int main()
{
    try
    {
        test_carrier();
        test_work_state_independent_of_text();
        test_call_errors();
        test_mismatch_and_abort();
        test_bootstrap_timeout();
        std::cout << "All error-code tests passed.\n";
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
