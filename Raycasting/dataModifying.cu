#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "light.h"

#include "dataModifying.cuh"
#include "constants.h"

__global__ void moveLights(float*, float*, float*, int, float, float);
__global__ void modifySpheresParameters(float3*, float*);


void modifyData(spheres* data)
{
	dim3 threadsInBlock(256, 1);
	dim3 blocks((sphere_count + 255) / 256, 1);

	modifySpheresParameters<<<blocks, threadsInBlock>>>(data->centers, data->radius);

	cudaDeviceSynchronize();
}


void moveLights(lights* lights)
{
	float tetha = 0.03f;

	moveLights<<<1, lights->count>>>(lights->x, lights->y, lights->z, lights->count, sinf(tetha), cosf(tetha));
}


/// <summary>
/// Kernel który modyfikuje wylosowane dane do parametrów podanych w constants.h
/// </summary>
/// <param name="centers">Tablica œrodków kul</param>
/// <param name="r">Tablica promieni kul</param>
__global__ void modifySpheresParameters(float3* centers, float* r)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= sphere_count) return;

	centers[x].x *= (MAX_X - MIN_X);
	centers[x].x += MIN_X;

	centers[x].y *= (MAX_Y - MIN_Y);
	centers[x].y += MIN_Y;

	centers[x].z *= (MAX_Z - MIN_Z);
	centers[x].z += MIN_Z;

	r[x] *= (MAX_RADIUS - MIN_RADIUS);
	r[x] += MIN_RADIUS;
}


/// <summary>
/// Kernel przemieszczaj¹cy wszystkie œwiat³a po okrêgu o podany k¹t
/// </summary>
/// <param name="x">Tablica wspó³rzêdnych x</param>
/// <param name="y">Tablica wspó³rzêdnych y</param>
/// <param name="z">Tablica wspó³rzêdnych z</param>
/// <param name="count">Liczba œwiate³</param>
/// <param name="sin_t">Sinus k¹ta obrotu</param>
/// <param name="cos_t">Cosinus k¹ta obrotu</param>
__global__ void moveLights(float* x, float* y, float* z, int count, float sin_t, float cos_t)
{
	int i = threadIdx.x;

	float xf = x[i], zf = z[i];

	x[i] = xf * cos_t - zf * sin_t;
	z[i] = xf * sin_t + zf * cos_t;
}
