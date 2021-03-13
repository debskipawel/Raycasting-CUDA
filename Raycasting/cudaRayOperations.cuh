#include <cuda_runtime.h>

#include "ray.h"
#include "sphere.h"
#include "hitReport.h"

/// <summary>
/// Funkcja wyliczaj�ca punkt w kt�rym znajduje si� promie�.
/// </summary>
/// <param name="r">Badany promie�</param>
/// <param name="t">Parametr t (r.origin + t * r.direction)</param>
/// <returns>Punkt na promieniu w danym "czasie".</returns>
__device__ float3 pointInTime(ray r, float t);

/// <summary>
/// Funkcja sprawdzaj�ca czy promie� uderzy� w podan� struktur� kul.
/// </summary>
/// <param name="spheres">Struktura kul</param>
/// <param name="r">Sprawdzany promie�</param>
/// <param name="t_min">Minimalne t jakie rozpatrujemy (parametr t w funkcji pointInTime).</param>
/// <param name="t_max">Maksymalne t jakie rozpatrujemy (parametr t w funkcji pointInTime).</param>
/// <returns>Informacje o najbli�szym uderzeniu.</returns>
__device__ hitReport checkSphereCollisions(
	spheres spheres,
	ray r,
	float t_min, float t_max
);