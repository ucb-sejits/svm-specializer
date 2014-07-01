#include <math.h>
#include <float.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif


cl_mem createBufferAndEnqueueWrite(cl_command_queue queue, cl_context context,
                                   cl_mem_flags flags, size_t size,
                                   const void *hostptr, cl_int *error_code_ret){
    cl_mem d_buffer = clCreateBuffer(context, flags, size, NULL, error_code_ret);
    if(!d_buffer){
        printf("Failed to create buffer!\n%d\n", *error_code_ret);
        return d_buffer;
    }
    if(hostptr != NULL){
        *error_code_ret = clEnqueueWriteBuffer(queue, d_buffer, CL_TRUE, 0, size, hostptr,0, NULL,NULL);
    }
    if (*error_code_ret != CL_SUCCESS){
        printf("Failed to enqueue write to buffer!\n%d\n", * error_code_ret);
        return d_buffer;
    }
    return d_buffer;
}

int train(float *input_data, int nPoints, int dFeatures){
    printf("Host Buffer Contents:\n");
    for(int i = 0; i < nPoints * dFeatures; i++){
        printf("input_data,%d,: %.2f\n", i,input_data[i]);
    }
    
    int err;                            // error code returned from api calls

    size_t globalFoph1;             // global domain size for our calculation

    size_t localFoph1;                  // local domain size for our calculation

    size_t vectorSize = sizeof(float) * dFeatures;

    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel foph1;                   // compute kernel

    // Set up platform and GPU device

    cl_uint numPlatforms;

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms <= 0){
        printf("Error: Failed to find a platform!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    if (err != CL_SUCCESS || numPlatforms <= 0){
        printf("Error: Failed to get the platform!\n");
        return EXIT_FAILURE;
    }

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++){
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL);
        if (err == CL_SUCCESS)
        {
            break;
        }
    }

    if (device_id == NULL){
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }
    
    // Create a compute context
    //
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context){
        printf("Error: Failed to create a compute context!\n");
        return err;
    }
    // Create a command commands
    //
    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands){
        printf("Error: Failed to create a command commands!\n");
        return err;
    }

    // Read the kernel into a string
    //
    FILE *kernelFile = fopen("kernel.cl", "rb");
    if (!kernelFile) {
        printf("Error: Coudn't open kernel file.\n");
    }

    fseek(kernelFile, 0L , SEEK_END);
    long len = ftell(kernelFile);
    rewind(kernelFile);

    // Allocate memory to hold kernel
    //
    char *KernelSource = calloc(1, len + 1);
    if (!KernelSource) {
        printf("Error: failed to allocate memory to hold kernel text.\n");
    }
    // Read the kernel into memory
    //
    if( 1 != fread(KernelSource, len, 1, kernelFile)) {
        printf("Error: failed to read file!\n");
    }

    fclose(kernelFile);
    // Create the compute program from the source buffer
    //
    program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
    if (!program){
        printf("Error: Failed to create compute program!\n");
        return err;
    }
    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[30000];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%d\n", err);
        printf("%s\n", buffer);
        return err;
    }

    // Create the compute kernel in the program we wish to run
    //
    foph1 = clCreateKernel(program, "firstOrderPhaseOne", &err);
    if (!foph1 || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        return err;
    }
    err = clGetKernelWorkGroupInfo(foph1, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localFoph1), &localFoph1, NULL);

    if (err != CL_SUCCESS){
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return err;
    }

    // Set the global work group sizes
    // Set to the first multiple of local size greater than nPoints
    int num_foph1_workgroups = (nPoints % localFoph1 != 0)?(nPoints / localFoph1 + 1):(nPoints/localFoph1);
    globalFoph1 = num_foph1_workgroups * localFoph1;

    cl_mem d_input_data = createBufferAndEnqueueWrite(commands, context, CL_MEM_READ_ONLY,
                                    vectorSize * nPoints, input_data, &err);
    if (err != CL_SUCCESS){
        printf("Error: Failed to allocate device memory!\n");
        return err;
    }

    err = 0;
    err  = clSetKernelArg(foph1, 0, sizeof(cl_mem), &d_input_data);
    err  |= clSetKernelArg(foph1, 1, sizeof(int), &nPoints);
    err  |= clSetKernelArg(foph1, 2, sizeof(int), &dFeatures);

    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }

    printf("Device Buffer Contents:\n");

    err = clEnqueueNDRangeKernel(commands, foph1, 1, NULL, &globalFoph1, &localFoph1, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to execute kernel!\n");
        return err;
    }
    // Wait for the command commands to get serviced before reading back results
    //
    err = clFinish(commands);

    if (err != CL_SUCCESS){
        printf("Error: waiting for queue to finish failed\n");
        return err;

    }
    // clean up
    clReleaseMemObject(d_input_data);
    clReleaseProgram(program);
    clReleaseKernel(foph1);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);
    return 0;
}
int main(){
    float input_data[50][2];
    for(int i = 0; i < 50; i++){
        for(int j=0; j <2; j++){
            input_data[i][j] = (float)(i + 1 - j);
        }
    }
    train(input_data, 50, 2);
    return 0;
}