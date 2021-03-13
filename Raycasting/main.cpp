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

// OpenGL Graphics includes
#include <helper_gl.h>
#include <GL/wglew.h>
#include <GL/freeglut.h>

// CUDA runtime
// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <helper_functions.h>
#include <helper_cuda.h>

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <chrono>

#include "parameters.h"
#include "dataModifying.cuh"
#include "graphicsKernels.h"
#include "cudaCameraRayOperations.cuh"
#include "dataGeneration.h"
#include "constants.h"

//OpenGL PBO and texture "names"
GLuint gl_PBO, gl_Tex, gl_Shader;
struct cudaGraphicsResource *cuda_pbo_resource; // handles OpenGL-CUDA exchange

//Source image on the host side
uchar4 *h_Src = 0;

// Destination image on the GPU side
uchar4 *d_dst = NULL;

int *pArgc = NULL;
char **pArgv = NULL;

#define REFRESH_DELAY 10

#define BUFFER_DATA(i) ((char *)0 + i)

void initGL(int*, char**);
void initOpenGLBuffers(int, int);
void renderImage();
void computeFPS();
void displayFunc(void);
void resizeFunc(int, int);
void keyboardFunc(unsigned char, int, int);
void cleanup();
void timerEvent(int);
GLuint compileASMShader(GLenum, const char*);

// gl_Shader for displaying floating-point texture
static const char* shader_code =
    "!!ARBfp1.0\n"
    "TEX result.color, fragment.texcoord, texture[0], 2D; \n"
    "END";

int main(int argc, char **argv)
{
    pArgc = &argc;
    pArgv = argv;

    // Inicjowanie parametrów programu w parameters.h
    initParameters();

    // Wype³nianie tablic constant memory (graphicsKernels.cu) losowymi kulami
    // oraz generowanie œwiate³ (œwiat³a równie¿ s¹ na pamiêci GPU, jedynie przechowywane s¹ tutaj)
    sendDataToConstantMemory();
    lightsStructure = generateLights(light_count);

    // Inicjalizacja OpenGLa
    initGL(&argc, argv);
    initOpenGLBuffers(imageW, imageH);

    glutCloseFunc(cleanup);

    glutMainLoop();
}

/// <summary>
/// Inicjalizacja okna OpenGL
/// </summary>
/// <param name="argc">WskaŸnik do liczby argumentów programu</param>
/// <param name="argv">Argumenty programu</param>
void initGL(int* argc, char** argv)
{
    printf("Initializing GLUT...\n");
    glutInit(argc, argv);

    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(imageW, imageH);
    glutInitWindowPosition(0, 0);
    
    char title[256];
    sprintf(title, "Raycasting (%d spheres, %d lights) - %d FPS", sphere_count, light_count, 0);
    
    glutCreateWindow(title);

    glutDisplayFunc(displayFunc);
    glutReshapeFunc(resizeFunc);
    glutKeyboardFunc(keyboardFunc);
    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);

    if (!isGLVersionSupported(1, 5) ||
        !areGLExtensionsSupported("GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
    {
        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
        fprintf(stderr, "This sample requires:\n");
        fprintf(stderr, "  OpenGL version 1.5\n");
        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
        exit(EXIT_SUCCESS);
    }

    printf("OpenGL window created.\n");
}

/// <summary>
/// Inicjalizacja buforów OpenGLa
/// </summary>
/// <param name="w">Szerokoœæ okna</param>
/// <param name="h">Wysokoœæ okna</param>
void initOpenGLBuffers(int w, int h)
{
    // delete old buffers
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    if (gl_Tex)
    {
        glDeleteTextures(1, &gl_Tex);
        gl_Tex = 0;
    }

    if (gl_PBO)
    {
        //DEPRECATED: checkCudaErrors(cudaGLUnregisterBufferObject(gl_PBO));
        cudaGraphicsUnregisterResource(cuda_pbo_resource);
        glDeleteBuffers(1, &gl_PBO);
        gl_PBO = 0;
    }

    // allocate new buffers
    h_Src = (uchar4*)malloc(w * h * sizeof(uchar4));

    printf("Creating GL texture...\n");
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &gl_Tex);
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, h_Src);
    printf("Texture created.\n");

    printf("Creating PBO...\n");
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, w * h * sizeof(uchar4), h_Src, GL_STREAM_COPY);

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
        cudaGraphicsMapFlagsWriteDiscard));
    printf("PBO created.\n");

    // load shader program
    gl_Shader = compileASMShader(GL_FRAGMENT_PROGRAM_ARB, shader_code);
}

/// <summary>
/// Funkcja renderuj¹ca teksturê wyœwietlan¹ póŸniej w oknie
/// </summary>
void renderImage()
{
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_dst, &num_bytes, cuda_pbo_resource));

    // wywo³anie kernela do obliczenia kolorów przy ka¿dym odœwie¿eniu tekstury
    calculateBackgroundColor(d_dst, lightsStructure, imageW, imageH, considerLightColor, ks, alpha);

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

/// <summary>
/// Funkcja wyœwietlaj¹ca obraz w oknie
/// </summary>
void displayFunc(void)
{
    // render the Mandelbrot image
    renderImage();

    // load texture from PBO
    glBindTexture(GL_TEXTURE_2D, gl_Tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, imageW, imageH, GL_RGBA, GL_UNSIGNED_BYTE, BUFFER_DATA(0));

    // fragment program is required to display floating point texture
    glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, gl_Shader);
    glEnable(GL_FRAGMENT_PROGRAM_ARB);
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, 0.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(0.0f, 1.0f);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_FRAGMENT_PROGRAM_ARB);

    glutSwapBuffers();

    computeFPS();
}

/// <summary>
/// Funkcja obliczaj¹ca iloœæ klatek na sekundê
/// </summary>
void computeFPS()
{
    static std::chrono::time_point<std::chrono::steady_clock> oldTime = std::chrono::high_resolution_clock::now();
    static int fps; fps++;

    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - oldTime) >= std::chrono::seconds{ 1 }) {
        oldTime = std::chrono::high_resolution_clock::now();
        char title[256];
        sprintf(title, "Raycasting (%d spheres, %d lights) - %d FPS", sphere_count, light_count, fps);
        glutSetWindowTitle(title);
        fps = 0;
    }
}

/// <summary>
/// Funkcja zmiany rozmiaru okna
/// </summary>
/// <param name="w">Nowa szerokoœæ okna</param>
/// <param name="h">Nowa wysokoœæ okna</param>
void resizeFunc(int w, int h)
{
    glViewport(0, 0, w, h);
    modifyAspectRatio(w, h);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

    if (w != 0 && h != 0)  // Do not call when window is minimized that is when width && height == 0
        initOpenGLBuffers(w, h);

    imageW = w;
    imageH = h;

    glutPostRedisplay();
}

/// <summary>
/// Obs³uga inputu u¿ytkownika z klawiatury
/// </summary>
/// <param name="ch">Wciœniêty klawisz</param>
void keyboardFunc(unsigned char ch, int, int)
{
    switch (ch)
    {
    case 'q': // obrót w lewo
    case 'Q':
        rotateCamera(-OMEGA);
        break;
    case 'e': // obrót w prawo
    case 'E':
        rotateCamera(OMEGA);
        break;
    case 'w': // ruch do przodu
    case 'W':
        moveCamera(FORWARD);
        break;
    case 'a': // ruch w lewo
    case 'A':
        moveCamera(LEFT);
        break;
    case 's': // ruch do ty³u
    case 'S':
        moveCamera(BACKWARD);
        break;
    case 'd': // ruch w prawo
    case 'D':
        moveCamera(RIGHT);
        break;
    case '-': // ruch w dó³
    case '_':
        moveCamera(DOWN);
        break;
    case '=': // ruch w górê
    case '+':
        moveCamera(UP);
        break;
    case '[': // zwiêkszenie liczby œwiate³
        if (light_count == 0) break;
        light_count--;
        lightsStructure = generateLights(light_count);
        break;
    case ']': // zmniejszenie liczby œwiate³
        if (light_count == 256) break;
        light_count++;
        lightsStructure = generateLights(light_count);
        break;
    case 'c': // zmiana ustawienia koloru œwiate³
    case 'C':
        considerLightColor = !considerLightColor;
        break;
    case 'm': // zwiêkszenie wspó³czynnika b³ysku
    case 'M':
        if (ks >= 1.0f) break;
        ks += 0.05f;
        break;
    case 'n': // zmniejszenie wspó³czynnika b³ysku
    case 'N':
        if (ks <= 0.0f) break;
        ks -= 0.05f;
        break;
    case 't': // w³¹czanie/wy³¹czanie obrotu œwiate³
    case 'T':
        areLightsMoving = !areLightsMoving;
        break;
    }
}

/// <summary>
/// Funkcja sprz¹taj¹ca bufory
/// </summary>
void cleanup()
{
    if (h_Src)
    {
        free(h_Src);
        h_Src = 0;
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glDeleteBuffers(1, &gl_PBO);
    glDeleteTextures(1, &gl_Tex);
    glDeleteProgramsARB(1, &gl_Shader);
}

/// <summary>
/// Funkcja wykonuj¹ca siê dla "tykniêcia" timera
/// </summary>
/// <param name="value"></param>
void timerEvent(int value)
{
    if (glutGetWindow())
    {
        if (areLightsMoving) moveLights(&lightsStructure);
        glutPostRedisplay();
        glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
    }
}

/// <summary>
/// Funkcja kompiluj¹ca shader OpenGLa
/// </summary>
/// <param name="program_type">Typ kompilowanego shadera</param>
/// <param name="code">Kod shadera</param>
/// <returns>Identyfikator skompilowanego shadera</returns>
GLuint compileASMShader(GLenum program_type, const char* code)
{
    GLuint program_id;
    glGenProgramsARB(1, &program_id);
    glBindProgramARB(program_type, program_id);
    glProgramStringARB(program_type, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte*)code);

    GLint error_pos;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &error_pos);

    if (error_pos != -1)
    {
        const GLubyte* error_string;
        error_string = glGetString(GL_PROGRAM_ERROR_STRING_ARB);
        fprintf(stderr, "Program error at position: %d\n%s\n", (int)error_pos, error_string);
        return 0;
    }

    return program_id;
}
