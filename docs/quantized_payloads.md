# Low-precision arithmetic vs opaque quantized payloads

TBCCL supports two different things, and keeps them apart on purpose.

| | A. Arithmetic reduction types | B. Opaque byte transport |
|---|---|---|
| What TBCCL does | combines two buffers element by element (`all_reduce`, `reduce`, `reduce_scatter`) | moves bytes unchanged (`send`/`recv`, `broadcast`, `all_gather`) |
| What it must understand | element type, element count, reduction op | nothing: `BufferView{kind, pointer, bytes}` |
| Public API shape | `BufferView` + `count` + `DataType` + `ReduceOp` | `BufferView` only (P2P also takes a size-check hint, see below) |
| Examples | Float16, BFloat16, Float32, Float64, Int8, UInt8, Int32, Int64 | FP8 (E4M3, E5M2, ...), INT8 weights, packed INT4/FP4 weights, scales, zero points, any future format |

`libtbccl` stays framework-neutral, model-neutral and quantization-format-neutral. It never contains the concepts GPTQ, AWQ, NF4, MXFP4, NVFP4, group size, zero point,
quantization scale or per-channel/per-group quantization. Those belong to the framework, the model runtime or the quantization library that owns the payload.

## Support matrix (Phase 49)

| Representation | Byte transport | SUM reduction | Notes |
|---|---|---|---|
| Float64, Float32, Int64, Int32 | yes | yes (all four `ReduceOp`s through the World API; `Communicator::all_reduce` is Sum only) | unchanged since Phase 41 |
| Float16 | yes | **yes, Sum** | widen to float32, one float32 add, round once, ties-to-even |
| BFloat16 | yes | **yes, Sum** | same, bfloat16 rounding (never truncation) |
| Int8, UInt8 | yes | **yes, Sum** | addition modulo 256 (two's complement for Int8); no signed overflow, no undefined behavior |
| FP8 E4M3 / E5M2 (and other FP8 variants) | **yes** | no | transport only; no reduction datatype exists, `all_reduce` is rejected before any communication |
| packed INT4 / FP4 (two values per byte) | **yes** | no | opaque packed bytes plus separate scale / zero-point buffers |
| anything else (random bytes) | yes | no | |

`reduction_supported(DataType, ReduceOp)` is the single predicate; `validate_reduction()` throws `unsupported: reduction dtype=<d> op=<o> (supported ops for this dtype: ...)` and every
public reducing entry point calls it before touching the network. `Capabilities::supports_collective_all_reduce(kind, dtype, op)` answers the same question per memory kind
(Host and Cuda: every supported type; MetalShared: the original four only).

## A. Reduction semantics

* **16-bit floats.** Values travel as raw 16-bit words. A SUM widens both operands to float32, performs one float32 add and rounds once to the target format with
  round-to-nearest-even. For N = 2 that is exactly one rounding per element, and it is what NCCL and PyTorch do for two operands. Host and CUDA implement it identically (the CUDA
  kernels use explicit conversions, not the native half/bfloat16 add, which rounds the exact sum once and could differ in rare double-rounding cases), so results agree
  **bit for bit**: verified for all 2^32 operand pairs of both formats. NaN results are NaN (payloads are not compared); infinities and overflow follow IEEE.
  Sequential algorithms with more than two ranks would re-round after each pairwise add; N > 2 low-precision semantics are not defined here.
* **Int8 / UInt8.** Sum modulo 256 through unsigned arithmetic (e.g. int8 120 + 100 = -36, 127 + 1 = -128, uint8 255 + 1 = 0). Product, Min and Max are not provided for the new types.
* **Alignment.** Buffers passed to a reduction must be aligned for the element type, as for Float32 today (tensors from frameworks are).
* **Endianness.** Nothing converts byte order. Both machines of the supported pair (x86-64 and arm64) are little-endian; a hypothetical big-endian peer would need its framework to convert.

## B. Transporting quantized payloads

Every opaque payload is one or more contiguous buffers. P2P `send`/`recv` still take `(count, datatype)`, but only as a size check (`count * datatype_size <= buffer.bytes`); the transfer moves
`BufferView::bytes`. Describe a payload as `DataType::UInt8` elements, one per byte, and no dtype is ever interpreted:

```cpp
// An AWQ/GPTQ-style layer, as the owning framework might hold it: packed 4-bit weights + per-group scales + zero points. TBCCL moves three independent buffers.
tbccl::BufferView packed{tbccl::MemoryKind::Host, packed_weights.data(), packed_weights.size(), -1};  // two 4-bit values per byte, low nibble first
tbccl::BufferView scales{tbccl::MemoryKind::Host, scale_words.data(),    scale_words.size(),    -1};  // 16-bit scale words, raw
tbccl::BufferView zeros {tbccl::MemoryKind::Host, zero_points.data(),    zero_points.size(),    -1};  // int8 zero points, raw
for (const tbccl::BufferView *v : {&packed, &scales, &zeros})
    comm.send(*v, v->bytes, tbccl::DataType::UInt8, /*peer=*/1).wait();
```

```python
# the same through torch.distributed (torch-tbccl): any dense dtype is byte-generic for send/recv, broadcast and all_gather
for t in (packed_uint8, scales_fp16, zeros_int8):
    dist.send(t, dst=1)
```

TBCCL does **not** know what the scales mean, how to dequantize, how groups are arranged, or that the weights are 4-bit. The receiver gets exactly the sender's bytes; the shape and layout
metadata (rows, columns, group size, packing order) is shared between the two framework processes by whatever means they already use. The same holds for FP8 words: an E4M3 and an E5M2 tensor are
both just bytes (NaN encodings and negative zero survive, because nothing is converted).

### Future formats (FP4, MXFP4, NVFP4, sub-byte formats)

The opaque approach needs no change for them: a packed FP4 tensor is a byte buffer, and an MX/NV block-scaled format is a payload buffer plus one or more scale buffers, transported as separate
buffers exactly like the INT4 bundle above, provided the owning framework transfers every metadata buffer the receiver needs. No special case is ever added to TBCCL. An arithmetic reduction on
such a format would need its own design (a separate storage-format concept, not an element-count `DataType`); sub-byte elements cannot be described by `count * datatype_size`.

## Where this is tested

* `tests/datatype_metadata_test.cpp`: enum values, sizes, labels, the support matrix, rejection before communication.
* `tests/reduction_low_precision_test.cpp`, `tests/reduction_int8_test.cpp`: host arithmetic (conversions proven by construction; all 65536 Int8/UInt8 pairs) and the real N=2 collectives.
* `tests/reduction_low_precision_cuda_test.cpp`: device == host bit for bit (all 2^32 pairs with `TBCCL_EXHAUSTIVE_LOWP=1`), Communicator paths CUDA<->Host both orientations and CUDA<->CUDA, delayed producer on a non-blocking stream.
* `tests/byte_transport_formats_test.cpp`: FP8 words, 16-bit/int8 buffers, a packed-int4 bundle with scales and zero points, and random bytes through send/recv, broadcast and all_gather, Host and CUDA.
* torch-tbccl `tests/test_lowprecision.py` and `tests/test_byte_transport.py`: the same through `torch.distributed` with real `torch.float8_e4m3fn` / `torch.float8_e5m2` tensors, packed-int4 bundles and every dtype this torch build provides.
