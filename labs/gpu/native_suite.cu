// This original teaching suite requires an NVIDIA GPU and the CUDA Toolkit.
// Compilation and execution are UNVERIFIED in the recorded CPU-only build.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::cerr << cudaGetErrorString(status) << " at line " << __LINE__ << '\n'; \
        std::exit(1); \
    } \
} while (0)

template<class T> T* allocate(std::size_t count) {
    T* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T)));
    return ptr;
}
void finish_launch() {
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
void check_close(float actual, float expected, const char* label) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > 1e-4f + 1e-4f * std::fabs(expected)) {
        std::cerr << label << ": actual=" << actual << ", expected=" << expected << '\n';
        std::exit(2);
    }
}
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
constexpr int TILE = 16;
__global__ void tiled_matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    __shared__ float aa[TILE][TILE];
    __shared__ float bb[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float total = 0.0f;
    for (int base = 0; base < k; base += TILE) {
        int ak = base + threadIdx.x;
        int bk = base + threadIdx.y;
        aa[threadIdx.y][threadIdx.x] = (row < m && ak < k) ? a[row * k + ak] : 0.0f;
        bb[threadIdx.y][threadIdx.x] = (bk < k && col < n) ? b[bk * n + col] : 0.0f;
        __syncthreads();
        for (int j = 0; j < TILE; ++j) total += aa[threadIdx.y][j] * bb[j][threadIdx.x];
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = total;
}
__constant__ float filter_coefficients[3];
__global__ void correlation_1d(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float total = 0.0f;
        for (int k = 0; k < 3; ++k) {
            int j = i + k - 1;
            if (j >= 0 && j < n) total += filter_coefficients[k] * x[j];
        }
        y[i] = total;
    }
}
__global__ void histogram(const unsigned* x, unsigned* bins, int n, int bin_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && x[i] < static_cast<unsigned>(bin_count)) atomicAdd(&bins[x[i]], 1u);
}
__global__ void reduce_blocks(const float* x, float* partial, int n) {
    __shared__ float sums[256];
    int lane = threadIdx.x;
    int i = blockIdx.x * blockDim.x + lane;
    int stride = blockDim.x * gridDim.x;
    float total = 0.0f;
    for (; i < n; i += stride) total += x[i];
    sums[lane] = total;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (lane < offset) sums[lane] += sums[lane + offset];
        __syncthreads();
    }
    if (lane == 0) partial[blockIdx.x] = sums[0];
}
int main() {
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { std::cerr << "An NVIDIA CUDA device is required.\n"; return 3; }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "device=" << prop.name << '\n';
    constexpr int n = 1003;
    std::vector<float> a(n), b(n), result(n);
    for (int i = 0; i < n; ++i) { a[i] = float(i % 11 - 5); b[i] = float(i % 7); }
    float* da = allocate<float>(n);
    float* db = allocate<float>(n);
    float* dc = allocate<float>(n);
    CUDA_CHECK(cudaMemcpy(da, a.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    vector_add<<<(n+255)/256,256>>>(da,db,dc,n); finish_launch();
    CUDA_CHECK(cudaMemcpy(result.data(),dc,n*sizeof(float),cudaMemcpyDeviceToHost));
    for (int i=0;i<n;++i) check_close(result[i],a[i]+b[i],"vector_add");
    std::cout << "vector_add=PASS\n";

    constexpr int m=37, inner=29, columns=35;
    std::vector<float> ma(m*inner), mb(inner*columns), mc(m*columns);
    for (int i=0;i<m*inner;++i) ma[i]=float(i%5-2);
    for (int i=0;i<inner*columns;++i) mb[i]=float(i%7-3);
    float* dma=allocate<float>(ma.size());
    float* dmb=allocate<float>(mb.size());
    float* dmc=allocate<float>(mc.size());
    CUDA_CHECK(cudaMemcpy(dma,ma.data(),ma.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dmb,mb.data(),mb.size()*sizeof(float),cudaMemcpyHostToDevice));
    tiled_matmul<<<dim3((columns+TILE-1)/TILE,(m+TILE-1)/TILE),dim3(TILE,TILE)>>>(dma,dmb,dmc,m,inner,columns);
    finish_launch();
    CUDA_CHECK(cudaMemcpy(mc.data(),dmc,mc.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for (int row=0;row<m;++row) for (int col=0;col<columns;++col) {
        float expected=0;
        for (int j=0;j<inner;++j) expected+=ma[row*inner+j]*mb[j*columns+col];
        check_close(mc[row*columns+col],expected,"tiled_matmul");
    }
    std::cout << "rectangular_tiled_matmul=PASS\n";

    float coeffs[3]={1.0f,2.0f,-1.0f};
    CUDA_CHECK(cudaMemcpyToSymbol(filter_coefficients,coeffs,sizeof(coeffs)));
    correlation_1d<<<(n+255)/256,256>>>(da,dc,n); finish_launch();
    CUDA_CHECK(cudaMemcpy(result.data(),dc,n*sizeof(float),cudaMemcpyDeviceToHost));
    for (int i=0;i<n;++i) {
        float expected=0;
        for (int k=0;k<3;++k) if (i+k-1>=0 && i+k-1<n) expected+=coeffs[k]*a[i+k-1];
        check_close(result[i],expected,"correlation");
    }
    std::cout << "zero_padded_correlation=PASS\n";

    constexpr int bin_count=8;
    std::vector<unsigned> keys(n), counts(bin_count), expected_counts(bin_count,0);
    for (int i=0;i<n;++i) { keys[i]=unsigned((i*i)%bin_count); ++expected_counts[keys[i]]; }
    unsigned* dkeys=allocate<unsigned>(n);
    unsigned* dbins=allocate<unsigned>(bin_count);
    CUDA_CHECK(cudaMemcpy(dkeys,keys.data(),n*sizeof(unsigned),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dbins,0,bin_count*sizeof(unsigned)));
    histogram<<<(n+255)/256,256>>>(dkeys,dbins,n,bin_count); finish_launch();
    CUDA_CHECK(cudaMemcpy(counts.data(),dbins,bin_count*sizeof(unsigned),cudaMemcpyDeviceToHost));
    if (counts!=expected_counts) { std::cerr << "histogram failed.\n"; return 2; }
    std::cout << "atomic_histogram=PASS\n";

    constexpr int blocks=4;
    float* dpartials=allocate<float>(blocks);
    std::vector<float> partials(blocks);
    reduce_blocks<<<blocks,256>>>(da,dpartials,n); finish_launch();
    CUDA_CHECK(cudaMemcpy(partials.data(),dpartials,blocks*sizeof(float),cudaMemcpyDeviceToHost));
    float total=0, reference=0;
    for (float v:partials) total+=v;
    for (float v:a) reference+=v;
    check_close(total,reference,"hierarchical_reduction");
    std::cout << "block_reduction_with_host_final_stage=PASS\n";
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    CUDA_CHECK(cudaFree(dma)); CUDA_CHECK(cudaFree(dmb)); CUDA_CHECK(cudaFree(dmc));
    CUDA_CHECK(cudaFree(dkeys)); CUDA_CHECK(cudaFree(dbins)); CUDA_CHECK(cudaFree(dpartials));
    return 0;
}
