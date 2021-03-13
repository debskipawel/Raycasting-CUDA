#include "constants.h"

#include <helper_math.h>

#include "cudaRayOperations.cuh"

__device__ float3 pointInTime(ray r, float t)
{
	return r.origin + r.direction * t;
}

__device__ hitReport checkSphereCollisions(
	spheres spheres, // spheres data
	ray r,
	float t_min, float t_max
)
{
	hitReport finalReport{ -1.0f };

	// zapisuje najbli¿sze uderzenie w kulê
	float closest_so_far = t_max;

	for (int i = 0; i < sphere_count; i++)
	{
		const sphere s
		{
			spheres.centers[i],
			spheres.radius[i]
		};

		// dla ka¿dej kuli s rozwi¹zuje równanie kwadratowe aby znaleŸæ dwa miejsca przeciêcia promienia z kul¹
		float3 oc = r.origin - s.center;

		float a = dot(r.direction, r.direction);
		float b = -dot(oc, r.direction);
		float c = dot(oc, oc) - s.radius * s.radius;

		float delta = b * b - a * c;

		if (delta <= 0) continue;

		float sqrtdelta = sqrtf(delta);

		// bierze mniejsze z wartoœci t, dla których promieñ uderza w kulê i porównuje z obecnie zapamiêtanym najmniejszym t
		float temp = fminf((b - sqrtdelta) / a, (b + sqrtdelta) / a);

		if (temp > t_min && temp < closest_so_far)
		{
			closest_so_far = temp;

			float3 hitPoint = pointInTime(r, temp);

			finalReport = hitReport
			{
				temp,
				normalize(hitPoint - s.center),
				hitPoint,
				spheres.colors[i]
			};
		}
	}

	return finalReport;
}