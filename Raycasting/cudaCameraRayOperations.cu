#include "cudaCameraRayOperations.cuh"

#include <vector_types.h>
#include <helper_math.h>

#include "constants.h"

/// <summary>
/// Lewy dolny róg symulowanego ekranu
/// </summary>
__device__ float3 lower_left = float3{ -2.0f, -1.0f, -1.0f };

/// <summary>
/// Poziomy wymiar symulowanego ekranu
/// </summary>
__device__ float3 horizontal = float3{ 4.0f, 0.0f, 0.0f };

/// <summary>
/// Pionowy wymiar symulowanego ekranu
/// </summary>
__device__ float3 vertical = float3{ 0.0f, 2.0f, 0.0f };

/// <summary>
/// Wspó³rzêdne kamery
/// </summary>
__device__ float3 origin = float3{ 0.0f, 0.0f, 0.0f };

/// <summary>
/// Kierunek w którym patrzy kamera
/// </summary>
__device__ float3 dir = float3{ 0.0f, 0.0f, -1.0f };

__device__ float3 unscaled_horizontal{ 4.0f, 0.0f, 0.0f };


__device__ ray getRay(float u, float v)
{
	return ray
	{
		origin,
		float3
		{
			lower_left.x + u * horizontal.x + v * vertical.x - origin.x,
			lower_left.y + u * horizontal.y + v * vertical.y - origin.y,
			lower_left.z + u * horizontal.z + v * vertical.z - origin.z
		}
	};
}


__device__ float3 getOrigin()
{
	return origin;
}


/// <summary>
/// Kernel modyfikuj¹cy aspect ratio ekranu.
/// </summary>
/// <param name="w">Szerokoœæ okna</param>
/// <param name="h">Wysokoœæ okna</param>
__global__ void modifyRatio(int w, int h)
{
	float ratio = float(w) / (2 * h - 80);

	float new_x = unscaled_horizontal.x * ratio;
	float new_z = unscaled_horizontal.z * ratio;

	float dx = horizontal.x - new_x;
	float dz = horizontal.z - new_z;

	horizontal.x = new_x;
	horizontal.z = new_z;
	
	lower_left.x += dx / 2;
	lower_left.z += dz / 2;
}

/// <summary>
/// Kernel przesuwaj¹cy kamerê.
/// </summary>
/// <param name="direction">Kierunek przesuniêcia</param>
__global__ void modifyCamera(int direction)
{
	float dx = 0.0f, dy = 0.0f, dz = 0.0f;

	switch (direction)
	{
	case UP:
		dy = 0.2f;
		break;
	case DOWN:
		dy = -0.2f;
		break;
	case RIGHT:
		dx = horizontal.x / 40;
		dz = horizontal.z / 40;
		break;
	case LEFT:
		dx = -horizontal.x / 40;
		dz = -horizontal.z / 40;
		break;
	case FORWARD:
		dx = dir.x / 10;
		dz = dir.z / 10;
		break;
	case BACKWARD:
		dx = - dir.x / 10;
		dz = - dir.z / 10;
		break;
	}

	origin.x += dx;
	lower_left.x += dx;
	origin.y += dy;
	lower_left.y += dy;
	origin.z += dz;
	lower_left.z += dz;
}

/// <summary>
/// Kernel obracaj¹cy kamerê.
/// </summary>
/// <param name="tetha">K¹t obrotu</param>
__global__ void modifyCameraAngle(float tetha)
{
	float3 LL = lower_left;
	float3 LR = lower_left + horizontal;

	float s = sinf(tetha);
	float c = cosf(tetha);

	// calculating new lower left corner
	float new_left_x = c * (LL.x - origin.x) - s * (LL.z - origin.z) + origin.x;
	float new_left_z = s * (LL.x - origin.x) + c * (LL.z - origin.z) + origin.z;

	// calculating new lower right corner (to calculate horizontal vector)
	float new_right_x = c * (LR.x - origin.x) - s * (LR.z - origin.z) + origin.x;
	float new_right_z = s * (LR.x - origin.x) + c * (LR.z - origin.z) + origin.z;

	float new_dir_x = c * dir.x - s * dir.z;
	float new_dir_z = s * dir.x + c * dir.z;

	// rotating the original on-scaled screen vector
	float unscaled_hor_x = c * unscaled_horizontal.x - s * unscaled_horizontal.z;
	float unscaled_hor_z = s * unscaled_horizontal.x + c * unscaled_horizontal.z;

	unscaled_horizontal.x = unscaled_hor_x;
	unscaled_horizontal.z = unscaled_hor_z;

	lower_left.x = new_left_x;
	lower_left.z = new_left_z;

	horizontal.x = new_right_x - new_left_x;
	horizontal.z = new_right_z - new_left_z;

	dir.x = new_dir_x;
	dir.z = new_dir_z;
}


void modifyAspectRatio(int width, int height)
{
	modifyRatio << <1, 1 >> > (width, height);
}


void moveCamera(int direction)
{
	modifyCamera << <1, 1 >> > (direction);
}


void rotateCamera(float tetha)
{
	modifyCameraAngle<<<1,1>>>(tetha);
}