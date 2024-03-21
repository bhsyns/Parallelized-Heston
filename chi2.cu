#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

__device__ float GS_star(curandState *state, float alpha) {
    float b = (alpha + expf(1.0)) / expf(1.0);
    float y, x, w;
    y = curand_uniform(state) * b;
    while (true)
    {
        if (y <= 1.0) {
        x = powf(y, 1.0 / alpha);
        w = -logf(curand_uniform(state));
        if (w > x) {
        return x;
        }
    } else {
        x = -logf((b - y) / alpha);
        w = powf(curand_uniform(state), 1.0 / (alpha - 1.0));
        if (w <= x) {
            return x;
        }
    }
    return 0.0; // Indicate failure or retry logic should be applied
    }
    
    
}

__device__ float GKM1(curandState *state, float alpha) {
    float a = alpha - 1.0f;
    float b = (alpha - 1.0f / (6.0f * alpha)) / a;
    float m = 2/a;
    float d = m + 2.0f;
    float v, x, y;

    while (true) {
        y = curand_uniform(state);
        x = curand_uniform(state);
        v = b * y / x;
        // Simplification of the accept/reject condition
        if (v+1/v + -d + m*x<= 0) {
            break;
        }
        if (m*logf(x)- log(v)+v-1 <= 0) {
            break;
        }
    }; // This condition is just a placeholder

    return a * v; // The actual GKM method will have a more complex calculation
}

__device__ float GKM2( curandState *state , float alpha) {
    float a = alpha - 1.0f;
    float b = (alpha - 1.0f / (6.0f * alpha)) / a;
    float m = 2/a;
    float d = m + 2.0f;
    float f = sqrtf(alpha);
    float v, x , x_prime , y;

    while (true) {
        while (true)
        {
            y = curand_uniform(state);
            x = curand_uniform(state);
            x_prime = y + (1-1.857764*x)/f ; 
            if (x_prime > 0 && x_prime < 1) {
                break;
            }
        }
        v = b * y / x_prime;
        if (v+1/v + -d + m*x_prime<= 0) {
            break;
        }
        if (m*logf(x_prime)- log(v)+v-1 <= 0) {
            break;
        }
    }; // This condition is just a placeholder

    return a * v; // The actual GKM method will have a more complex calculation


}

__device__ float GKM3(curandState *state, float alpha) {
    float alpha_0 = 2.5f;

    if (alpha < alpha_0) {
        return GKM1(state, alpha);
    } else {
        return GKM2(state, alpha);
    }
}

__device__ float generate_gamma(curandState *state, float alpha) {
    if (alpha < 1.0f) {
        return GS_star(state, alpha);
    } else {
        return GKM3(state, alpha);
    }
}


__device__ float chi2(curandState *state, int d) {
    float alpha = d / 2.0f;
    float scale = 2.0f; // Scale factor for chi-square from gamma.
    return generate_gamma(state, alpha) * scale;

}

__global__ void non_central_chi2_kernel(float *results, int d, float lambda_, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(1234, idx, 0, &state);

        float alpha = d / 2.0f;
        float N = curand_poisson(&state, lambda_ / 2.0f);
        
        float chi2_val = 2.0f * generate_gamma(&state, alpha + N);  // Gamma(d/2, 2)
        
        results[idx] = chi2_val;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Usage: %s <d> <n> <lambda>\n", argv[0]);
        return 1; // Indicate error
    }

    int d = atoi(argv[1]);
    int n = atoi(argv[2]);
    float lambda_ = atof(argv[3]);

    if(d <= 0 || n <= 0 || lambda_ < 0) {
        printf("Error: Ensure that d > 0, n > 0, and lambda >= 0.\n");
        return 1;
    }

    int NTPB = 1024; // Threads per block
    int NB = (n + NTPB - 1) / NTPB; // Number of blocks

    float *results;

    cudaMallocManaged(&results, n * sizeof(float));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    non_central_chi2_kernel<<<NB, NTPB>>>(results, d, lambda_, n);
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    for (int i = 0; i < n; i++) {
        printf("%f\n", results[i]);
    }
    
    printf("Time: %f\n", duration.count());
    
    cudaFree(results);
    return 0;
}