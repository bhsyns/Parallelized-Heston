/**************************************************************
Based on BLACK SCHOLES code by Lokman A. Abbas-Turki

Those who re-use this code should mention in their code
the name of the author above.
***************************************************************/

#include <stdio.h>
#include <curand_kernel.h>
#include <stdlib.h>


// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

/*One-Dimensional Normal Law. Cumulative distribution function. */
double NP(double x) {
	const double p = 0.2316419;
	const double b1 = 0.319381530;
	const double b2 = -0.356563782;
	const double b3 = 1.781477937;
	const double b4 = -1.821255978;
	const double b5 = 1.330274429;
	const double one_over_twopi = 0.39894228;
	double t;

	if (x >= 0.0) {
		t = 1.0 / (1.0 + p * x);
		return (1.0 - one_over_twopi * exp(-x * x / 2.0) * t * (t * (t *
			(t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0 / (1.0 - p * x);
		return (one_over_twopi * exp(-x * x / 2.0) * t * (t * (t * (t *
			(t * b5 + b4) + b3) + b2) + b1));
	}
}

__global__ void init_curand_state_k(curandState *state) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}
// Heston model Monte Carlo simulation kernel
__global__ void MC_k(float S_0, float r, float kappa, float theta, float sigma_v, float rho, float dt, float K, int N, curandState* state, float* sum, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	curandState localState = state[idx];
	float S = S_0;
	extern __shared__ float A[];

	float* R1s, * R2s;
	R1s = A;
	R2s = R1s + blockDim.x;

  float V = theta; // Initial volatility

	 for (int i = 0; i < N; i++) {
        float2 G = curand_normal2(&localState);
        float dW_S = G.x; // Brownian motion increment for asset price
        float dW_V = rho * G.x + sqrtf(1 - rho * rho) * G.y; // Brownian motion increment for volatility
        float vol = sqrtf(fmaxf(0, V)); // Ensure volatility is non-negative
        S += r * S * dt * dt + vol * S * dW_S * dt; // Update asset price
        V += kappa * (theta - V) * dt *dt + sigma_v * vol * dt* dW_V;
    }


	float payoff = fmaxf(0.0f, S - K);
	R1s[threadIdx.x] = expf(-r * dt * dt * N) * payoff / n;
	R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x] * n;

	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0) {
		if (threadIdx.x < i) {
			R1s[threadIdx.x] += R1s[threadIdx.x + i];
			R2s[threadIdx.x] += R2s[threadIdx.x + i];
		}
		__syncthreads();
		i /= 2;
	}

	if (threadIdx.x == 0) {
		atomicAdd(sum, R1s[0]);
		atomicAdd(sum + 1, R2s[0]);
	}
	/* copy state back to global memory*/
	// state[idx] = localState;  
}


int main(int argc, char *argv[]) {
    // Check if the correct number of arguments are provided
    if (argc != 11) {
        printf("Usage: ./MC <S_0> <K> <r> <kappa> <theta> <sigma_v> <rho> <T> <N> <M>\n");
        return 1;
    }

    // Convert command-line arguments to appropriate data types
    float S_0 = atof(argv[1]);
    float K = atof(argv[2]);
    float r = atof(argv[3]);
    float kappa = atof(argv[4]);
    float theta = atof(argv[5]);
    float sigma_v = atof(argv[6]);
    float rho = atof(argv[7]);
    float T = atof(argv[8]);
    int N = atoi(argv[9]);
    int M = atoi(argv[10]);

    // Constants and variables for the simulation
    int NTPB = 1024; // Threads per block
    int NB = (M + NTPB - 1) / NTPB; // Number of blocks, round up if necessary
    int n = NB * NTPB; // Total number of threads
    float dt = sqrtf(T / N);
    float *sum;

    // Allocate memory for storing results on the GPU
    cudaMallocManaged(&sum, 2 * sizeof(float));
    cudaMemset(sum, 0, 2 * sizeof(float));

    // Allocate memory for random number generator states on the GPU
    curandState* states;
    cudaMalloc(&states, n * sizeof(curandState));

    // Initialize random number generator states
    init_curand_state_k<<<NB, NTPB>>>(states);

    // GPU timer instructions
    float Tim;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Monte Carlo simulation kernel call
    MC_k<<<NB, NTPB, 2 * NTPB * sizeof(float)>>>(S_0, r, kappa, theta, sigma_v, rho, dt, K, N, states, sum, n);

    // GPU timer instructions
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&Tim, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print results
    printf("The estimated price is equal to %f\n", sum[0]);
    printf("error associated to a confidence interval of 95%% = %f\n",
           1.96 * sqrt((double)(sum[1] - (sum[0] * sum[0]))) / sqrt((double)n));
    // Adjust calculation of true price according to Heston model
    printf("Execution time %f ms\n", Tim);

    // Free allocated memory on the GPU
    cudaFree(sum);
    cudaFree(states);

    return 0;
}