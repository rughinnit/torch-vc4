/*
 * torch_vc4 - a deliberately tiny PyTorch PrivateUse1 backend for the
 * Raspberry Pi VideoCore IV, running on top of libvc4infer over vc4d.
 *
 * Design rules (kept deliberately narrow and honest):
 *   - device "vc4" == c10::DeviceType::PrivateUse1, one device, index 0;
 *   - every tensor lives inside one region of the engine's dma-buf arena,
 *     obtained through one vc4_alloc() and sub-allocated here;
 *   - only these aten ops exist on vc4:
 *       empty.memory_format, empty_strided, copy_, mm, matmul (2-D),
 *       add.Tensor, mul.Tensor, relu, normal_
 *     everything else hits a catch-all PrivateUse1 fallback that throws.
 *     There is no CPU fallback path anywhere in this file.
 *   - all engine calls are synchronous (the daemon's LAUNCH_WAIT round trip),
 *     so there is no stream/queue and no async error surface.
 *
 * This file deliberately avoids torch/extension.h (the C++ frontend plus
 * pybind11): a 1 GB board cannot compile it.  Only ATen + torch/library.h
 * are used, and the Python-visible helpers are registered as a small custom
 * operator namespace (torch.ops.torch_vc4.*).
 */
#include <ATen/ATen.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "vc4infer.h"

namespace torch_vc4 {

constexpr c10::DeviceType kDev = c10::DeviceType::PrivateUse1;
constexpr c10::DeviceIndex kDevIndex = 0;
constexpr size_t kAlign = 64;      /* TMU/DMA lines */
constexpr size_t kSlack = 64;      /* kernel over-read/write guard */
constexpr size_t kReserveWant = 24u << 20;   /* of the 64 MiB client arena */
constexpr size_t kReserveMin = 4u << 20;

/* ------------------------------------------------------------------ state */

struct Block {
    size_t off;      /* offset inside the torch region */
    size_t size;
    bool used;
};

struct State {
    vc4_ctx *ctx = nullptr;
    uint8_t *base = nullptr;     /* host mapping of the torch region */
    uint32_t region_off = 0;     /* region offset inside the arena */
    size_t cap = 0;              /* region size in bytes */
    std::vector<Block> blocks;
    std::mutex gpu_mu;           /* vc4_ctx is not thread safe */
    std::mutex alloc_mu;
    std::mt19937 rng{std::random_device{}()};
};

/* Leaked on purpose: tensors may outlive static destruction at exit. */
static State &state()
{
    static State *s = new State();
    return *s;
}

static const char *default_cache_dir()
{
    const char *e = getenv("VC4INFER_CACHE");
    if (e && *e)
        return e;
    return "/home/raspi/work/engine-daemon/kcache";
}

static std::string init_failure;

static void ensure_init()
{
    static std::once_flag once;
    std::call_once(once, [] {
        State &s = state();
        /* Never touch /dev/mem from this process: force the vc4d backend
         * unless the caller explicitly chose otherwise. */
        if (!getenv("VC4INFER_DAEMON"))
            setenv("VC4INFER_DAEMON", "1", 0);

        vc4_ctx *ctx = nullptr;
        vc4_status st = vc4_init(&ctx, default_cache_dir(), 0);
        if (st != VC4_OK) {
            init_failure = std::string("vc4_init failed: ") + vc4_strerror(st);
            return;
        }
        s.ctx = ctx;

        size_t want = kReserveWant;
        vc4_buf region{0, 0};
        for (;;) {
            st = vc4_alloc(ctx, want, &region);
            if (st == VC4_OK || want <= kReserveMin)
                break;
            want /= 2;
        }
        if (st != VC4_OK) {
            init_failure = std::string("vc4_alloc(") +
                           std::to_string(kReserveMin) + " B) failed: " +
                           vc4_strerror(st) + " (arena bytes " +
                           std::to_string((unsigned long long)
                                              vc4_arena_bytes(ctx)) + ")";
            return;
        }
        s.base = (uint8_t *)vc4_host_ptr(ctx, &region);
        s.region_off = region.off;
        s.cap = region.bytes;
        s.blocks.push_back(Block{0, s.cap, false});
    });
    if (!state().ctx)
        TORCH_CHECK(false, "torch_vc4: ", init_failure);
}

static void throw_st(vc4_status st, const char *what)
{
    std::string msg = std::string("torch_vc4: ") + what + ": " +
                      vc4_strerror(st);
    const char *detail = vc4_last_error(state().ctx);
    if (detail && *detail)
        msg += std::string(" (") + detail + ")";
    TORCH_CHECK(false, msg);
}

/* ----------------------------------------------------------------- arena */

static size_t arena_alloc(size_t bytes)
{
    ensure_init();
    State &s = state();
    std::lock_guard<std::mutex> g(s.alloc_mu);
    size_t want = (bytes + kSlack + kAlign - 1) & ~(kAlign - 1);
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        Block &b = s.blocks[i];
        if (b.used || b.size < want)
            continue;
        /* NB: capture the offset before insert() - it can reallocate the
         * vector and invalidate `b`. */
        const size_t off = b.off;
        if (b.size == want) {
            b.used = true;
        } else {
            Block rest{off + want, b.size - want, false};
            b.size = want;
            b.used = true;
            s.blocks.insert(s.blocks.begin() + i + 1, rest);
        }
        if (getenv("TORCH_VC4_DEBUG"))
            fprintf(stderr, "[vc4] alloc %zu -> off %zu (base %p)\n", bytes,
                    off, (void *)(s.base + off));
        return off;
    }
    TORCH_CHECK(false, "torch_vc4: out of arena: requested ", want,
                " B, region ", (unsigned long long)s.cap,
                " B is full (torch region only; the rest belongs to the "
                "engine's scratch)");
}

static void arena_free(size_t off)
{
    State &s = state();
    std::lock_guard<std::mutex> g(s.alloc_mu);
    for (size_t i = 0; i < s.blocks.size(); ++i) {
        if (s.blocks[i].off != off)
            continue;
        TORCH_INTERNAL_ASSERT(s.blocks[i].used);
        s.blocks[i].used = false;
        while (i + 1 < s.blocks.size() && !s.blocks[i + 1].used) {
            s.blocks[i].size += s.blocks[i + 1].size;
            s.blocks.erase(s.blocks.begin() + i + 1);
        }
        if (i > 0 && !s.blocks[i - 1].used) {
            s.blocks[i - 1].size += s.blocks[i].size;
            s.blocks.erase(s.blocks.begin() + i);
        }
        return;
    }
    TORCH_INTERNAL_ASSERT(false, "torch_vc4: freeing unknown arena offset");
}

/* The DataPtr context is left null on purpose: c10 stores the data pointer as
 * the context in that case, so the deleter always receives a data pointer and
 * the arena offset is recovered by subtraction. */
static void arena_deleter(void *p)
{
    State &s = state();
    ptrdiff_t off = static_cast<char *>(p) - reinterpret_cast<char *>(s.base);
    TORCH_INTERNAL_ASSERT(off >= 0 && (size_t)off < s.cap,
                          "torch_vc4: free of a pointer outside the arena");
    arena_free((size_t)off);
}

struct Vc4Allocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t n) override
    {
        size_t off = arena_alloc(n ? n : 1);
        State &s = state();
        void *p = s.base + off;
        return c10::DataPtr(p, nullptr, &arena_deleter,
                            c10::Device(kDev, kDevIndex));
    }
    c10::DeleterFnPtr raw_deleter() const override { return &arena_deleter; }
    void copy_data(void *dest, const void *src, size_t count) const override
    {
        /* Byte copy of two arena addresses; never used to compute an op. */
        memmove(dest, src, count);
    }
};

/* A vc4_buf describing a tensor that lives in the arena. */
static vc4_buf buf_of(const at::Tensor &t)
{
    State &s = state();
    const char *p = static_cast<const char *>(t.const_data_ptr());
    ptrdiff_t local = p - reinterpret_cast<const char *>(s.base);
    size_t bytes = (size_t)t.numel() * t.element_size();
    TORCH_CHECK(local >= 0 && (size_t)local + bytes <= s.cap,
                "torch_vc4: tensor storage is not inside the vc4 arena "
                "(ptr ", (void *)p, " base ", (void *)s.base, " local ",
                (long long)local, " bytes ", bytes, " cap ", s.cap, ")");
    return vc4_buf{(uint32_t)(s.region_off + (size_t)local), bytes};
}

/* ------------------------------------------------------------------ guard */

struct Vc4GuardImpl final : public c10::impl::DeviceGuardImplInterface {
    static constexpr c10::DeviceType kType = c10::DeviceType::PrivateUse1;

    Vc4GuardImpl() = default;
    explicit Vc4GuardImpl(c10::DeviceType t)
    {
        TORCH_INTERNAL_ASSERT(t == kType);
    }

    c10::DeviceType type() const override { return kType; }

    c10::Device exchangeDevice(c10::Device d) const override
    {
        TORCH_CHECK(d.type() == kType && d.index() == kDevIndex,
                    "torch_vc4: invalid vc4 device ", d);
        return d;
    }
    c10::Device getDevice() const override
    {
        return c10::Device(kType, kDevIndex);
    }
    void setDevice(c10::Device d) const override
    {
        TORCH_CHECK(d.type() == kType && d.index() == kDevIndex,
                    "torch_vc4: invalid vc4 device ", d);
    }
    void uncheckedSetDevice(c10::Device) const noexcept override {}

    c10::Stream getStream(c10::Device) const noexcept override
    {
        return c10::Stream(c10::Stream::UNSAFE,
                           c10::Device(kType, kDevIndex), 0);
    }
    c10::Stream exchangeStream(c10::Stream) const noexcept override
    {
        return c10::Stream(c10::Stream::UNSAFE,
                           c10::Device(kType, kDevIndex), 0);
    }
    c10::DeviceIndex deviceCount() const noexcept override { return 1; }

    /* Every launch is a synchronous LAUNCH_WAIT round trip. */
    void synchronizeDevice(c10::DeviceIndex) const override {}
    void synchronizeStream(const c10::Stream &) const override {}
    bool queryStream(const c10::Stream &) const override { return true; }
};

/* ------------------------------------------------------------------ empty */

static void check_vc4_f32(const at::Tensor &t, const char *what)
{
    TORCH_CHECK(t.device().type() == kDev, "torch_vc4: ", what,
                ": expected a vc4 tensor, got ", t.device());
    TORCH_CHECK(t.scalar_type() == at::kFloat, "torch_vc4: ", what,
                ": only float32 is supported, got ", t.scalar_type());
}

static at::Tensor make_arena_tensor(c10::IntArrayRef sizes)
{
    int64_t numel = 1;
    for (auto s : sizes) {
        TORCH_CHECK(s >= 0, "torch_vc4: negative size");
        numel *= s;
    }
    size_t bytes = (size_t)numel * sizeof(float);
    auto dp = c10::GetAllocator(kDev)->allocate(bytes ? bytes : 1);
    auto *holder = new c10::DataPtr(std::move(dp));
    auto opts = at::TensorOptions()
                    .dtype(at::kFloat)
                    .device(c10::Device(kDev, kDevIndex));
    return at::from_blob(holder->get(), sizes,
                         [holder](void *) { delete holder; }, opts);
}

static at::Tensor vc4_empty(c10::IntArrayRef size,
                            std::optional<c10::ScalarType> dtype,
                            std::optional<c10::Layout> layout,
                            std::optional<c10::Device> device,
                            std::optional<bool> pin_memory,
                            std::optional<c10::MemoryFormat> memory_format)
{
    TORCH_CHECK(!device || device->type() == kDev,
                "torch_vc4: empty called for device ",
                device ? device->str() : std::string("<none>"));
    TORCH_CHECK(!dtype || *dtype == at::kFloat,
                "torch_vc4: only float32 tensors are supported, got ",
                dtype ? c10::toString(*dtype) : "float32");
    TORCH_CHECK(!layout || *layout == at::kStrided,
                "torch_vc4: only strided layout is supported");
    TORCH_CHECK(!pin_memory || !*pin_memory,
                "torch_vc4: pinned memory is not supported");
    TORCH_CHECK(!memory_format ||
                    *memory_format == at::MemoryFormat::Contiguous,
                "torch_vc4: only contiguous memory format is supported");
    return make_arena_tensor(size);
}

static at::Tensor vc4_empty_strided(c10::IntArrayRef size,
                                    c10::IntArrayRef stride,
                                    std::optional<c10::ScalarType> dtype,
                                    std::optional<c10::Layout> layout,
                                    std::optional<c10::Device> device,
                                    std::optional<bool> pin_memory)
{
    /* Only accept strides that describe a contiguous tensor: our tensors are
     * packed, and pretending to support arbitrary strides would silently
     * produce wrong results. */
    int64_t expect = 1;
    bool contiguous = true;
    for (int64_t i = (int64_t)size.size() - 1; i >= 0; --i) {
        if (size[i] == 0)
            continue;
        if (stride[i] != expect) {
            contiguous = false;
            break;
        }
        expect *= size[i];
    }
    TORCH_CHECK(contiguous,
                "torch_vc4: empty_strided: only contiguous strides are "
                "supported on vc4");
    return vc4_empty(size, dtype, layout, device, pin_memory,
                     at::MemoryFormat::Contiguous);
}

/* ------------------------------------------------------------------- copy */

static at::Tensor &vc4_copy_(at::Tensor &self, const at::Tensor &src,
                             bool non_blocking)
{
    (void)non_blocking;
    ensure_init();
    const bool self_vc4 = self.device().type() == kDev;
    const bool src_vc4 = src.device().type() == kDev;
    TORCH_CHECK(self_vc4 || src_vc4,
                "torch_vc4: copy_ called with neither side on vc4");
    TORCH_CHECK(self.sizes() == src.sizes(),
                "torch_vc4: copy_ needs identical shapes (got ", self.sizes(),
                " and ", src.sizes(), "); broadcasting is not implemented");
    TORCH_CHECK(self.scalar_type() == src.scalar_type(),
                "torch_vc4: copy_ does not convert dtypes (",
                self.scalar_type(), " vs ", src.scalar_type(), ")");
    TORCH_CHECK(self.scalar_type() == at::kFloat,
                "torch_vc4: only float32 is supported");

    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    size_t bytes = (size_t)self.numel() * sizeof(float);

    if (src_vc4 && !self_vc4) {
        TORCH_CHECK(self.is_contiguous(),
                    "torch_vc4: copy_ to a non-contiguous CPU tensor is not "
                    "supported");
        vc4_buf b = buf_of(src);
        vc4_status st = vc4_download(s.ctx, &b, self.mutable_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ vc4->cpu");
        return self;
    }
    if (!src_vc4 && self_vc4) {
        at::Tensor c = src.is_contiguous() ? src : src.contiguous();
        vc4_buf b = buf_of(self);
        vc4_status st = vc4_upload(s.ctx, &b, c.const_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ cpu->vc4");
        return self;
    }
    /* vc4 -> vc4: stage through the host so the engine's dirty tracking and
     * cache maintenance stay in charge (byte copy, no arithmetic). */
    {
        at::Tensor c = src.is_contiguous() ? src : src.contiguous();
        vc4_buf b = buf_of(self);
        vc4_status st = vc4_upload(s.ctx, &b, c.const_data_ptr(), bytes);
        if (st != VC4_OK)
            throw_st(st, "copy_ vc4->vc4");
    }
    return self;
}

/* --------------------------------------------------------------------- mm */

static at::Tensor vc4_mm(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    check_vc4_f32(a, "mm(A)");
    check_vc4_f32(b, "mm(B)");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: mm needs 2-D tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: mm needs contiguous tensors "
                "(call .contiguous() first)");
    int64_t M = a.size(0), K = a.size(1), K2 = b.size(0), N = b.size(1);
    TORCH_CHECK(K == K2, "torch_vc4: mm shape mismatch ", a.sizes(), " x ",
                b.sizes());

    auto out = make_arena_tensor({M, N});
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bc = buf_of(out);
    vc4_status st = vc4_matmul(s.ctx, &ba, &bb, &bc, (uint32_t)M, (uint32_t)N,
                               (uint32_t)K, VC4_LAYOUT_ROW_MAJOR);
    if (st != VC4_OK)
        throw_st(st, "matmul");
    return out;
}

static at::Tensor vc4_matmul(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "torch_vc4: matmul is implemented for 2-D x 2-D only (got ",
                a.dim(), "-D x ", b.dim(), "-D)");
    return vc4_mm(a, b);
}

/* ----------------------------------------------------------- binary/unary */

static void check_same_shape(const at::Tensor &a, const at::Tensor &b,
                             const char *what)
{
    TORCH_CHECK(a.sizes() == b.sizes(), "torch_vc4: ", what,
                " requires identical shapes, got ", a.sizes(), " and ",
                b.sizes(), "; broadcasting is not implemented");
}

static at::Tensor vc4_add(const at::Tensor &a, const at::Tensor &b,
                          const at::Scalar &alpha)
{
    ensure_init();
    check_vc4_f32(a, "add");
    check_vc4_f32(b, "add");
    check_same_shape(a, b, "add");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: add needs contiguous tensors");
    TORCH_CHECK(alpha.toDouble() == 1.0,
                "torch_vc4: add with alpha != 1 is not implemented (got ",
                alpha.toDouble(), ")");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s.ctx, VC4_OP_ADD, &ba, &bb, &bo,
                                    (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "add");
    return out;
}

static at::Tensor vc4_mul(const at::Tensor &a, const at::Tensor &b)
{
    ensure_init();
    check_vc4_f32(a, "mul");
    check_vc4_f32(b, "mul");
    check_same_shape(a, b, "mul");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "torch_vc4: mul needs contiguous tensors");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bb = buf_of(b), bo = buf_of(out);
    vc4_status st = vc4_elementwise(s.ctx, VC4_OP_MUL, &ba, &bb, &bo,
                                    (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "mul");
    return out;
}

static at::Tensor vc4_relu(const at::Tensor &a)
{
    ensure_init();
    check_vc4_f32(a, "relu");
    TORCH_CHECK(a.is_contiguous(), "torch_vc4: relu needs a contiguous tensor");
    auto out = make_arena_tensor(a.sizes());
    State &s = state();
    std::lock_guard<std::mutex> g(s.gpu_mu);
    vc4_buf ba = buf_of(a), bo = buf_of(out);
    vc4_status st = vc4_relu(s.ctx, &ba, &bo, (size_t)a.numel());
    if (st != VC4_OK)
        throw_st(st, "relu");
    return out;
}

/* ----------------------------------------------------------- normal_ (RNG) */
/* Random numbers are generated on the host (std::mt19937) and uploaded.  The
 * GPU has no RNG kernel; this is stated in the README rather than hidden. */
static at::Tensor &vc4_normal_(at::Tensor &self, double mean, double std,
                               std::optional<at::Generator> generator)
{
    (void)generator;
    ensure_init();
    check_vc4_f32(self, "normal_");
    TORCH_CHECK(self.is_contiguous(),
                "torch_vc4: normal_ needs a contiguous tensor");
    size_t n = (size_t)self.numel();
    std::vector<float> host(n);
    {
        State &s = state();
        std::lock_guard<std::mutex> g(s.alloc_mu);
        const char *seed = getenv("TORCH_VC4_SEED");
        if (seed && *seed) {
            s.rng.seed((uint32_t)strtoul(seed, nullptr, 0));
            unsetenv("TORCH_VC4_SEED");
        }
        std::normal_distribution<float> dist((float)mean, (float)std);
        for (size_t i = 0; i < n; ++i)
            host[i] = dist(s.rng);
    }
    std::lock_guard<std::mutex> g(state().gpu_mu);
    vc4_buf b = buf_of(self);
    vc4_status st = vc4_upload(state().ctx, &b, host.data(),
                               n * sizeof(float));
    if (st != VC4_OK)
        throw_st(st, "normal_");
    return self;
}

/* ---------------------------------------------------------------- fallback */

static void vc4_fallback(const c10::OperatorHandle &op, c10::DispatchKeySet,
                         torch::jit::Stack *)
{
    TORCH_CHECK(false, "torch_vc4: '", op.schema().operator_name().name,
                "' is not implemented for device vc4.  This backend has no "
                "CPU fallback by design; supported ops: empty, empty_strided, "
                "copy_/to, mm, matmul(2-D), add, mul, relu, normal_.");
}

/* ------------------------------------------------------------------- misc */

/* ---------------------------------------------------------------- hooks */
/* torch requires a PrivateUse1HooksInterface before a privateuse1 device can
 * be used at all ("Please register PrivateUse1HooksInterface ... first"). */
struct Vc4Hooks final : public at::PrivateUse1HooksInterface {
    bool isBuilt() const override { return true; }
    bool isAvailable() const override { return true; }
    bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
    c10::DeviceIndex deviceCount() const override { return 1; }
    c10::DeviceIndex getCurrentDevice() const override { return 0; }
    void setCurrentDevice(c10::DeviceIndex) const override {}

    const at::Generator &getDefaultGenerator(c10::DeviceIndex) const override
    {
        /* torch only ever asks for a generator to pass to normal_-style
         * kernels; ours ignores it and uses the host std::mt19937. */
        static at::Generator gen = at::detail::createCPUGenerator(42u);
        return gen;
    }

    at::Device getDeviceFromPtr(void *) const override
    {
        /* There is exactly one vc4 device, so every pointer torch asks about
         * here belongs to it (this is also called for pointers that are not
         * arena allocations, e.g. while a factory op is being set up). */
        return at::Device(kDev, kDevIndex);
    }

    void resizePrivateUse1Bytes(const c10::Storage &, size_t) const override
    {
        TORCH_CHECK(false, "torch_vc4: resizing vc4 tensor storage is not "
                           "supported (the arena allocator has no realloc)");
    }
};

/* Kick the hooks/allocator in at library load time. */
static void register_backend()
{
    static std::once_flag once;
    std::call_once(once, [] {
        c10::SetAllocator(kDev, new Vc4Allocator());
        at::RegisterPrivateUse1HooksInterface(new Vc4Hooks());
    });
}

}  // namespace torch_vc4

/* ------------------------------------------------------------ registration */

static struct AllocatorRegistrar {
    AllocatorRegistrar() { torch_vc4::register_backend(); }
} allocator_registrar;

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
    m.impl("empty.memory_format", TORCH_FN(torch_vc4::vc4_empty));
    m.impl("empty_strided", TORCH_FN(torch_vc4::vc4_empty_strided));
    m.impl("copy_", TORCH_FN(torch_vc4::vc4_copy_));
    m.impl("mm", TORCH_FN(torch_vc4::vc4_mm));
    m.impl("matmul", TORCH_FN(torch_vc4::vc4_matmul));
    m.impl("add.Tensor", TORCH_FN(torch_vc4::vc4_add));
    m.impl("mul.Tensor", TORCH_FN(torch_vc4::vc4_mul));
    m.impl("relu", TORCH_FN(torch_vc4::vc4_relu));
    m.impl("normal_", TORCH_FN(torch_vc4::vc4_normal_));
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m)
{
    m.fallback(torch::CppFunction::makeFromBoxedFunction<
               &torch_vc4::vc4_fallback>());
}

C10_REGISTER_GUARD_IMPL(PrivateUse1, torch_vc4::Vc4GuardImpl);

/* Diagnostics, reachable as torch.ops.torch_vc4.stats() etc.  (No pybind11:
 * it is not worth the compile memory on this board.  The returns are plain
 * double lists; torch_vc4/__init__.py turns them into named dicts.) */
TORCH_LIBRARY(torch_vc4, m)
{
    m.def("stats", []() -> std::vector<double> {
        torch_vc4::ensure_init();
        vc4_stats st{};
        vc4_get_stats(torch_vc4::state().ctx, &st);
        return {(double)st.launches,
                (double)st.pack_ns / 1e6,
                (double)st.kernel_ns / 1e6,
                (double)st.gpu_cycles,
                (double)st.flops,
                st.kernel_ns ? (double)st.flops / (double)st.kernel_ns : 0.0};
    });
    m.def("reset_stats", []() {
        torch_vc4::ensure_init();
        vc4_reset_stats(torch_vc4::state().ctx);
    });
    m.def("seed", [](int64_t s) {
        torch_vc4::ensure_init();
        std::lock_guard<std::mutex> g(torch_vc4::state().alloc_mu);
        torch_vc4::state().rng.seed((uint32_t)s);
    });
    m.def("arena", []() -> std::vector<double> {
        torch_vc4::ensure_init();
        torch_vc4::State &s = torch_vc4::state();
        return {(double)s.region_off,
                (double)s.cap,
                (double)vc4_arena_bytes(s.ctx),
                (double)vc4_arena_used(s.ctx)};
    });
}
