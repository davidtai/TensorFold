// W2 tool (w2/proto-qr1, never merged; qsa_q4 with QR-1's arms; coordinator 2026-10-29, "KV in 4-token blocks"):
// QR-1 alone (V in 4-token blocks, a dim's 4 tokens together, p.v's B fragments straight from the cache), Q4-2 alone (K
// in 4-token blocks of 16-dim groups) and both, against the shipped QSA kernels, from engine builds' gfx1151 code
// objects, over a TF_W4_PICKS dump. Every arm's bytes are compared with the shipped kernel's.
//   prompt: tf_fn_sparse_prompt_hi (main) against w4_sparse_ql_ref (the copy), _ql10 (K coalesced), _ql3 (V in QL-3's
//           layout), _ql3k (both); the engine's launch, grid (rows, 2), 128 threads, gated bf16 out.
//   decode tile: tf_fn_sparse_tile_parts (main) against w4_sparse_tile_parts_ref and _ql3 over the dump's last 1 and
//           last 8 rows (a decode row, a verify window), 32 parts, as fn_sparse_decode launches it.
//   kv_write: tf_kv_write (main ops) against w4_kv_write_ql3 for the prompt's rows x 2 kv heads x 256 bf16; the QL-3
//           cache it writes is checked against the host's layout of the same values.
// Usage: qsa_qr1 reps PICKS MAIN_FLASHNEXT.hsaco PROTO_FLASHNEXT.hsaco MAIN_OPS.hsaco
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static uint16_t bf(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}
static float fb(uint16_t h) {
    uint32_t u = static_cast<uint32_t>(h) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}
#define CK(e) do { hipError_t r_ = (e); if (r_ != hipSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #e, hipGetErrorString(r_)); exit(1); } } while (0)

template <typename T>
static T* dev(size_t n) {
    T* p;
    CK(hipMalloc(&p, n * sizeof(T)));
    return p;
}
template <typename T>
static T* up(const std::vector<T>& v) {
    T* p = dev<T>(v.size());
    CK(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return p;
}
template <typename T>
static std::vector<T> down(const T* p, size_t n) {
    std::vector<T> v(n);
    CK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}
template <typename T>
static size_t differ(const std::vector<T>& a, const std::vector<T>& b) {
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) n += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    return n;
}
static hipFunction_t fn(const char* path, const char* sym) {
    hipModule_t m;
    CK(hipModuleLoad(&m, path));
    hipFunction_t f;
    CK(hipModuleGetFunction(&f, m, sym));
    return f;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: qsa_qr1 reps PICKS MAIN_FN.hsaco PROTO_FN.hsaco MAIN_OPS.hsaco\n"); return 2; }
    const int reps = atoi(argv[1]);
    FILE* f = fopen(argv[2], "rb");
    if (!f) { perror(argv[2]); return 1; }
    int head[3];
    if (fread(head, 4, 3, f) != 3) return 1;
    const int rows = head[0], stride = head[1];
    std::vector<int> count(rows), sparse(rows), ids(static_cast<size_t>(rows) * stride);
    if (fread(count.data(), 4, rows, f) != static_cast<size_t>(rows) || fread(sparse.data(), 4, rows, f) != static_cast<size_t>(rows) ||
        fread(ids.data(), 4, ids.size(), f) != ids.size()) { fprintf(stderr, "short dump\n"); return 1; }
    fclose(f);
    const long long cap = 20544;  // a multiple of 4 (QL-3's blocks)
    uint32_t st = 0x12345u;
    auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (st >> 8) * (1.f / 16777216.f) - 0.5f; };
    std::vector<float> hq(static_cast<size_t>(rows) * 24 * 256);
    for (auto& x : hq) x = fb(bf(rnd() * 4.f));
    std::vector<uint16_t> hk(static_cast<size_t>(2) * cap * 256), hv(hk.size()), hg(static_cast<size_t>(rows) * 24 * 512);
    for (auto& x : hk) x = bf(rnd() * 2.f);
    for (auto& x : hv) x = bf(rnd() * 2.f);
    for (auto& x : hg) x = bf(rnd() * 3.f);
    // QL-3's V: dword ((g cap / 4 + key / 4) 128 + pair) 4 + key % 4 holds dims (2 pair, 2 pair + 1)
    std::vector<uint16_t> hv4(hv.size());
    for (long long g = 0; g < 2; ++g)
        for (long long key = 0; key < cap; ++key)
            for (int d = 0; d < 256; ++d)
                hv4[(((g * (cap / 4) + key / 4) * 128 + d / 2) * 4 + key % 4) * 2 + d % 2] = hv[(g * cap + key) * 256 + d];
    // QR-1's V: element ((g cap / 4 + key / 4) 256 + d) 4 + key % 4
    std::vector<uint16_t> hvb(hv.size());
    for (long long g = 0; g < 2; ++g)
        for (long long key = 0; key < cap; ++key)
            for (int d = 0; d < 256; ++d)
                hvb[((g * (cap / 4) + key / 4) * 256 + d) * 4 + key % 4] = hv[(g * cap + key) * 256 + d];
    // Q4-2's K: element ((g cap / 4 + key / 4) 16 + d / 16) 64 + (key % 4) 16 + d % 16
    std::vector<uint16_t> hk4(hk.size());
    for (long long g = 0; g < 2; ++g)
        for (long long key = 0; key < cap; ++key)
            for (int d = 0; d < 256; ++d)
                hk4[((g * (cap / 4) + key / 4) * 16 + d / 16) * 64 + (key % 4) * 16 + d % 16] = hk[(g * cap + key) * 256 + d];
    float* q = up(hq);
    uint16_t* k4 = up(hk4);
    uint16_t *k = up(hk), *v = up(hv), *v4 = up(hv4), *vbf = up(hvb), *qg = up(hg);
    int *di = up(ids), *dc = up(count), *ds = up(sparse);
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    // median ms a launch of `go` (each sample: `inner` launches back to back)
    auto timeit = [&](auto&& go, int inner) {
        go();
        CK(hipDeviceSynchronize());
        std::vector<float> t;
        for (int i = 0; i < reps; ++i) {
            CK(hipEventRecord(a));
            for (int j = 0; j < inner; ++j) go();
            CK(hipEventRecord(b));
            CK(hipEventSynchronize(b));
            float ms;
            CK(hipEventElapsedTime(&ms, a, b));
            t.push_back(ms / inner);
        }
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    long long id_stride = stride;
    float scale = 0.0625f;
    int fail = 0;

    // ---- prompt tile
    {
        const size_t out_n = static_cast<size_t>(rows) * 24 * 256;
        uint16_t* out = dev<uint16_t>(out_n);
        struct Arm { const char* path; const char* sym; int vl; bool kblk; };  // vl: 0 rows, 1 QL-3, 2 QR-1
        const Arm arms[] = {{argv[3], "tf_fn_sparse_prompt_hi", 0, false}, {argv[4], "w4_sparse_ql_ref", 0, false},
                            {argv[4], "w2_sparse_q42", 0, true},           {argv[4], "w2_sparse_qr1", 2, false},
                            {argv[4], "w2_sparse_qr12", 2, true}};
        std::vector<uint16_t> ref;
        float t0 = 0.f;
        for (const Arm& arm : arms) {
            const hipFunction_t fx = fn(arm.path, arm.sym);
            uint16_t* vv = arm.vl == 2 ? vbf : arm.vl == 1 ? v4 : v;
            uint16_t* kk = arm.kblk ? k4 : k;
            void* args[] = {&q, &kk, &vv, &di, &id_stride, &dc, &ds, &qg, &out, (void*)&cap, &scale};
            CK(hipMemset(out, 0xff, out_n * 2));
            const float ms = timeit([&] { CK(hipModuleLaunchKernel(fx, rows, 2, 1, 128, 1, 1, 0, nullptr, args, nullptr)); }, 1);
            auto o = down(out, out_n);
            if (ref.empty()) { ref = o; t0 = ms; }
            const size_t dif = differ(ref, o);
            fail |= dif != 0;
            printf("prompt %-24s rows %d: %.3f ms (x%.3f), bytes %zu of %zu differ%s\n", arm.sym, rows, ms, t0 / ms, dif, out_n,
                   dif ? " FAIL" : "");
        }
        CK(hipFree(out));
    }

    // ---- decode tile: the dump's last `nr` rows, 32 parts
    for (int nr : {1, 2, 4, 8}) {
        const int parts = 32, r0 = rows - nr;
        const size_t on = static_cast<size_t>(nr) * 24 * parts * 256, mn = static_cast<size_t>(nr) * 24 * parts * 2;
        float *po = dev<float>(on), *pm = dev<float>(mn);
        float* qr = q + static_cast<size_t>(r0) * 24 * 256;
        int *ir = di + static_cast<size_t>(r0) * stride, *cr = dc + r0, *sr = ds + r0;
        struct Arm { const char* path; const char* sym; int vl; bool kblk; };
        const Arm arms[] = {{argv[3], "tf_fn_sparse_tile_parts", 0, false}, {argv[4], "w4_sparse_tile_parts_ref", 0, false},
                            {argv[4], "w2_sparse_tile_parts_q42", 0, true}, {argv[4], "w2_sparse_tile_parts_qr1", 2, false},
                            {argv[4], "w2_sparse_tile_parts_qr12", 2, true}};
        std::vector<float> ro, rm;
        float t0 = 0.f;
        for (const Arm& arm : arms) {
            const hipFunction_t fx = fn(arm.path, arm.sym);
            uint16_t* vv = arm.vl == 2 ? vbf : arm.vl == 1 ? v4 : v;
            uint16_t* kk = arm.kblk ? k4 : k;
            void* args[] = {&qr, &kk, &vv, &ir, &id_stride, &cr, &sr, &po, &pm, (void*)&cap, &scale};
            CK(hipMemset(po, 0xff, on * 4));
            const float ms = timeit([&] { CK(hipModuleLaunchKernel(fx, nr, 2, parts, 128, 1, 1, 0, nullptr, args, nullptr)); }, 50);
            auto o = down(po, on), m = down(pm, mn);
            if (ro.empty()) { ro = o; rm = m; t0 = ms; }
            const size_t dif = differ(ro, o) + differ(rm, m);
            fail |= dif != 0;
            printf("decode %-26s rows %d (keys %d..): %.4f ms (x%.3f, %+.4f ms), bytes %zu of %zu differ%s\n", arm.sym, nr,
                   count[r0], ms, t0 / ms, ms - t0, dif, on + mn, dif ? " FAIL" : "");
        }
        CK(hipFree(po));
        CK(hipFree(pm));
    }

    // ---- kv_write of the prompt's values (rows x 2 kv heads x 256) at slot 0
    {
        const int len = rows, kvh = 2, d = 256, kind = 2, total = static_cast<int>(cap), pos0 = 0;
        const size_t n = static_cast<size_t>(len) * kvh * d;
        std::vector<uint16_t> src(n);
        for (auto& x : src) x = bf(rnd() * 2.f);
        uint16_t* s = up(src);
        uint16_t *c_row = dev<uint16_t>(hv.size()), *c_blk = dev<uint16_t>(hv.size());
        CK(hipMemset(c_row, 0, hv.size() * 2));
        CK(hipMemset(c_blk, 0, hv.size() * 2));
        const hipFunction_t fr = fn(argv[5], "tf_kv_write"), fq = fn(argv[4], "w4_kv_write_ql3"), fk = fn(argv[4], "w2_kv_write_kblk");
        const unsigned blocks = static_cast<unsigned>((n + 255) / 256);
        void* ar[] = {&s, &c_row, (void*)&kind, (void*)&len, (void*)&kvh, (void*)&d, (void*)&total, (void*)&pos0};
        void* aq[] = {&s, &c_blk, (void*)&len, (void*)&kvh, (void*)&total, (void*)&pos0};
        const float tr = timeit([&] { CK(hipModuleLaunchKernel(fr, blocks, 1, 1, 256, 1, 1, 0, nullptr, ar, nullptr)); }, 10);
        const float tq = timeit([&] { CK(hipModuleLaunchKernel(fq, blocks, 1, 1, 256, 1, 1, 0, nullptr, aq, nullptr)); }, 10);
        const auto row = down(c_row, hv.size()), blk = down(c_blk, hv.size());
        size_t bad = 0;
        for (long long g = 0; g < 2; ++g)
            for (long long key = 0; key < cap; ++key)
                for (int dd = 0; dd < 256; ++dd)
                    bad += blk[(((g * (cap / 4) + key / 4) * 128 + dd / 2) * 4 + key % 4) * 2 + dd % 2] != row[(g * cap + key) * 256 + dd];
        fail |= bad != 0;
        printf("kv_write rows %d: tf_kv_write %.4f ms, w4_kv_write_ql3 %.4f ms (%+.4f ms); QL-3 cache against the row cache: %zu of %zu differ%s\n",
               len, tr, tq, tq - tr, bad, hv.size(), bad ? " FAIL" : "");
        uint16_t* c_kb = dev<uint16_t>(hv.size());
        CK(hipMemset(c_kb, 0, hv.size() * 2));
        void* ak[] = {&s, &c_kb, (void*)&len, (void*)&kvh, (void*)&total, (void*)&pos0};
        const float tk = timeit([&] { CK(hipModuleLaunchKernel(fk, blocks, 1, 1, 256, 1, 1, 0, nullptr, ak, nullptr)); }, 10);
        const auto kb = down(c_kb, hv.size());
        size_t badk = 0;
        for (long long g = 0; g < 2; ++g)
            for (long long key = 0; key < cap; ++key)
                for (int dd = 0; dd < 256; ++dd)
                    badk += kb[((g * (cap / 4) + key / 4) * 16 + dd / 16) * 64 + (key % 4) * 16 + dd % 16] != row[(g * cap + key) * 256 + dd];
        fail |= badk != 0;
        printf("kv_write rows %d: w2_kv_write_kblk %.4f ms (%+.4f ms); Q4-2 cache against the row cache: %zu of %zu differ%s\n",
               len, tk, tk - tr, badk, hv.size(), badk ? " FAIL" : "");
        const hipFunction_t fv = fn(argv[4], "w2_kv_write_vbf");
        uint16_t* c_vb = dev<uint16_t>(hv.size());
        CK(hipMemset(c_vb, 0, hv.size() * 2));
        void* av[] = {&s, &c_vb, (void*)&len, (void*)&kvh, (void*)&total, (void*)&pos0};
        const float tv = timeit([&] { CK(hipModuleLaunchKernel(fv, blocks, 1, 1, 256, 1, 1, 0, nullptr, av, nullptr)); }, 10);
        const auto vbh = down(c_vb, hv.size());
        size_t badv = 0;
        for (long long g = 0; g < 2; ++g)
            for (long long key = 0; key < cap; ++key)
                for (int dd = 0; dd < 256; ++dd)
                    badv += vbh[((g * (cap / 4) + key / 4) * 256 + dd) * 4 + key % 4] != row[(g * cap + key) * 256 + dd];
        fail |= badv != 0;
        printf("kv_write rows %d: w2_kv_write_vbf %.4f ms (%+.4f ms); QR-1 cache against the row cache: %zu of %zu differ%s\n",
               len, tv, tv - tr, badv, hv.size(), badv ? " FAIL" : "");
    }
    printf("QSA_QR1 %s\n", fail ? "FAIL" : "ok");
    return fail;
}
