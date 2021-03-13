#pragma once

#include <vector_types.h>

/// <summary>
/// Struktura do zwracania wyniku uderzenia promienia w kulê.
/// </summary>
struct hitReport
{
	float t;			// parametr t (hitPoint = ray.origin + t * ray.direction)
	float3 normal;		// wektor normalny w punkcie zderzenia
	float3 hitPoint;	// punkt zderzenia
	float3 color;		// kolor kuli
};