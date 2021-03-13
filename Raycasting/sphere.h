#pragma once

#include "ray.h"

/// <summary>
/// Struktura zawieraj�ca �rodek kuli, jej promie� i kolor
/// </summary>
struct sphere 
{
	float3 center;	// �rodek kuli
	float radius;	// promie� kuli
	float3 color;	// kolor kuli
};

/// <summary>
/// Struktura zawieraj�ca �rodki kul, ich promienie i kolory, w tablicach
/// </summary>
struct spheres
{
	float3* centers;
	float* radius;
	float3* colors;
};