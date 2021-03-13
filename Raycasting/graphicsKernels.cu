/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

//#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "helper_cuda.h"

#include <vector_types.h>
#include <helper_math.h>

#include "graphicsKernels.h"
#include "math.h"

#include "cudaRayOperations.cuh"
#include "cudaCameraRayOperations.cuh"

#include "dataGeneration.h"
#include "constants.h"
#include "ray.h"
#include "light.h"

__constant__ float3 centers[sphere_count];
__constant__ float radius[sphere_count];
__constant__ float3 colors[sphere_count];

__device__ float3 cudaBackgroundColor(ray, lights, bool, float, float);
__global__ void backgroundColorKernel(uchar4*, lights, int, int, bool, float, float);

void calculateBackgroundColor(uchar4* dev, lights lights, int w, int h, bool considerLightColor, float ks, float alpha)
{
    dim3 threadsInBlock(16, 16);
    dim3 blocks((w + 16 - 1) / 16, (h + 16 - 1) / 16);

    backgroundColorKernel<<<blocks, threadsInBlock>>> (dev, lights, w, h, considerLightColor, ks, alpha);
}

void sendDataToConstantMemory()
{
    // generuje dane
    spheres s = generateSpheres();

    // przenosi je do pamiêci constant
    checkCudaErrors(cudaMemcpyToSymbol(centers, s.centers, sphere_count * sizeof(float3), 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(radius, s.radius, sphere_count * sizeof(float), 0, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(colors, s.colors, sphere_count * sizeof(float3), 0, cudaMemcpyDeviceToDevice));

    // zwalnia zaalokowan¹ pamiêæ
    checkCudaErrors(cudaFree(s.centers));
    checkCudaErrors(cudaFree(s.radius));
    checkCudaErrors(cudaFree(s.colors));
}


/// <summary>
/// Kernel wyliczaj¹cy wielow¹tkowo kolor ka¿dego piksela tekstury o podanym rozmiarze.
/// </summary>
/// <param name="output">Wynikowa tablica pikseli (ich kolorów)</param>
/// <param name="lights">Struktura œwiate³</param>
/// <param name="width">Szerokoœæ okna</param>
/// <param name="height">Wysokoœæ okna</param>
/// <param name="considerLightColor">Czy œwiat³a maj¹ oœwietlaæ na bia³o, czy na swoje kolory</param>
__global__ void backgroundColorKernel(uchar4* output, lights lights, int width, int height, bool considerLightColor, float ks, float alpha)
{
    // oblicza wspó³rzêdne i je¿eli jest poza tekstur¹, koñczy
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    float u = float(x) / width;
    float v = float(y) / height;

    // oblicza promieñ w danym punkcie ekranu
    ray r = getRay(u, v);

    // oblicza kolor piksela i ustawia outputu
    float3 color = cudaBackgroundColor(r, lights, considerLightColor, ks, alpha);

    output[y * width + x] = uchar4
    {
        (unsigned char)(color.x * 255),
        (unsigned char)(color.y * 255),
        (unsigned char)(color.z * 255),
        (unsigned char)255
    };
}


/// <summary>
/// Funkcja zwracaj¹ca kolor, jaki odpowiada danemu promieniowi.
/// </summary>
/// <param name="ray">Œledzony promieñ</param>
/// <param name="lights">Struktura œwiate³</param>
/// <param name="considerLightColor">Czy œwiat³a maj¹ oœwietlaæ na bia³o, czy na swoje kolory</param>
/// <returns>Kolor piksela</returns>
__device__ float3 cudaBackgroundColor(ray ray, lights lights, bool considerLightColor, float ks, float alpha)
{
    spheres sph
    {
        centers,
        radius,
        colors
    };

    // zdobywa informacje o najbli¿szej kuli w któr¹ uderza œledzony promieñ
    hitReport report = checkSphereCollisions(sph, ray, 0.001f, 1000.0f);

    // jeœli znaleziono jak¹œ kulê, przechodzi po œwiat³ach i stosuje
    // wzór z modelu Phonga dla ka¿dego œwiat³a
    if (report.t > 0.0f)
    {
        float3 color{ 0.0f, 0.0f, 0.0f };

        for (int i = 0; i < lights.count; i++)
        {
            float3 lightPos{ lights.x[i], lights.y[i], lights.z[i] };
            float3 lightColor = considerLightColor ? float3{ lights.r[i], lights.g[i], lights.b[i] } : float3{ 1.0f, 1.0f, 1.0f };

            // wektor od punktu uderzenia do œwiat³a
            float3 light = normalize(lightPos - report.hitPoint);

            float dot_ln = fmaxf(dot(light, report.normal), 0.0f);

            // znormalizowany wektor od punktu uderzenia do kamery
            float3 V = normalize(getOrigin() - report.hitPoint);
            // znormalizowany wektor odbicia
            float3 R = normalize(((2.0f * dot_ln) * report.normal) - light);

            float dot_rv = dot(R, V);

            color += lightColor * report.color * ((1.0f - ks) * dot_ln + ks * powf(dot_rv, alpha));
        }

        // obcinanie do poprawnych kolorów RGB
        return clamp(color, 0.0f, 1.0f);
    }

    // jeœli promieñ nie przecina kuli, zwraca kolor nieba
    float3 unitDir = normalize(ray.direction);

    float t = 0.5 * (unitDir.y + 1.0);

    return float3
    {
        1.0f - 0.5f * t,
        1.0f - 0.3f * t,
        1.0f
    };
}
