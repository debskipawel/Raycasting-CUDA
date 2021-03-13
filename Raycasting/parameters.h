#pragma once
#include "light.h"

int imageW, imageH;

lights lightsStructure;
int light_count;

bool considerLightColor;
bool areLightsMoving;

int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
float avgFPS = 0.0f;
unsigned int frameCount = 0;
StopWatchInterface* timer = NULL;

float ks;
float alpha = 1.0f;

void initParameters()
{
	imageW = 700;
	imageH = 440;
	
	lightsStructure = {};
	light_count = 2;
	
	considerLightColor = false;
	areLightsMoving = true;
	
	ks = 0.5f; 

	sdkCreateTimer(&timer);
}