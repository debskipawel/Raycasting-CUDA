#include "constants.h"

#include <curand.h>
#include <iostream>
#include <ctime>

#include "helper_cuda.h"

#include "dataGeneration.h"
#include "dataModifying.cuh"

spheres generateSpheres()
{
	spheres data;

	// alokuje pamiêæ na GPU
	checkCudaErrors(cudaMalloc(&(data.centers), sphere_count * sizeof(float3)));
	checkCudaErrors(cudaMalloc(&(data.radius), sphere_count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&(data.colors), sphere_count * sizeof(float3)));

	// tworzy generator cuRand
	curandGenerator_t gen;
	curandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	curandCheck(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	// losuje wartoœci przy u¿yciu biblioteki cuRand
	curandCheck(curandGenerateUniform(gen, (float*)data.centers, 3 * sphere_count));
	curandCheck(curandGenerateUniform(gen, data.radius, sphere_count));
	curandCheck(curandGenerateUniform(gen, (float*)data.colors, 3 * sphere_count));

	// niszczy generator po zakoñczeniu losowania
	curandCheck(curandDestroyGenerator(gen));

	// modyfikuje wylosowane dane
	modifyData(&data);

	// zwraca gotowe dane
	return data;
}

lights generateLights(int count)
{
	if (count == 0) return lights{};

	lights lights;
	lights.count = count;

	constexpr float radius = 100.0f;

	// alokuje pamiêæ GPU na œwiat³a
	checkCudaErrors(cudaMalloc(&(lights.x), count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&(lights.y), count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&(lights.z), count * sizeof(float)));

	checkCudaErrors(cudaMalloc(&(lights.r), count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&(lights.g), count * sizeof(float)));
	checkCudaErrors(cudaMalloc(&(lights.b), count * sizeof(float)));

	// tworzy generator cuRand
	curandGenerator_t gen;
	curandCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	curandCheck(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

	// alokuje pamiêæ hosta na pozycje œwiate³
	float* x = new float[count];
	float* y = new float[count];
	float* z = new float[count];

	// sekwencyjnie wype³nia kolejne œwiat³a, rozmieszczaj¹c je równomiernie na okrêgu o promieniu 100
	x[0] = 0.0f; y[0] = 0.0f; z[0] = radius;

	const float sin_d_phi = sinf(6.28f / count);
	const float cos_d_phi = cosf(6.28f / count);

	for (int i = 1; i < count; i++)
	{
		x[i] = x[i - 1] * cos_d_phi - z[i - 1] * sin_d_phi;
		y[i] = 0.0f;
		z[i] = x[i - 1] * sin_d_phi + z[i - 1] * cos_d_phi;
	}

	// kopiuje wype³nione dane do pamiêci GPU
	checkCudaErrors(cudaMemcpy(lights.x, x, count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(lights.y, y, count * sizeof(float), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(lights.z, z, count * sizeof(float), cudaMemcpyHostToDevice));

	// usuwa bufory ze strony hosta
	delete x;
	delete y;
	delete z;

	// losuje kolory œwiate³
	curandCheck(curandGenerateUniform(gen, lights.r, count));
	curandCheck(curandGenerateUniform(gen, lights.g, count));
	curandCheck(curandGenerateUniform(gen, lights.b, count));

	// niszczy generator cuRand
	curandCheck(curandDestroyGenerator(gen));

	return lights;
}