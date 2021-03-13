#pragma once

#include "ray.h"

/// <summary>
/// Struktura zawieraj¹ca œrodek kuli, jej promieñ i kolor
/// </summary>
struct sphere 
{
	float3 center;	// œrodek kuli
	float radius;	// promieñ kuli
	float3 color;	// kolor kuli
};

/// <summary>
/// Struktura zawieraj¹ca œrodki kul, ich promienie i kolory, w tablicach
/// </summary>
struct spheres
{
	float3* centers;
	float* radius;
	float3* colors;
};