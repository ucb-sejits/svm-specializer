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

// Kernel Functions
float linearSelf(float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float value = vecA[i];
        accumulant += value * value;
    }
    return accumulant;
}
float linear(float *vecA, float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        accumulant += vecA[i] * vecB[i];
    }
    return accumulant;
}
float gaussianSelf(float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    return 1.0f;
}
float gaussian(float *vecA, float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float diff = vecA[i] - vecB[i];
        accumulant += diff * diff;
    }
    return exp(- paramA * accumulant);
}
float polynomialSelf(float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float value = vecA[i];
        accumulant += value * value;
    }
    accumulant = accumulant * paramA + paramB;
    float result = accumulant;
    for (float degree = 2.0f; degree <= paramC; degree = degree + 1.0f) {
        result *= accumulant;
    }
    return result;
}
float polynomial(float *vecA, float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        accumulant += vecA[i] * vecB[i];
    }
    accumulant = accumulant * paramA + paramB;
    float result = accumulant;
    for (float degree = 2.0f; degree <= paramC; degree = degree + 1.0f) {
        result *= accumulant;
    }
    return result;
}
float sigmoidSelf(float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float value = vecA[i];
        accumulant += value * value;
    }
    accumulant = accumulant * paramA + paramB;
    return tanh(accumulant);
}
float sigmoid(float *vecA, float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        accumulant += vecA[i] * vecB[i];
    }
    accumulant = accumulant * paramA + paramB;
    return tanh(accumulant);
}

int train(float *input_data, float *labels, float *training_alpha,
            float epsilon, float Ce, float cost, float tolerance,
            int heuristic, int nPoints,  int dFeatures,
            float paramA, float paramB, float paramC, float *trainResult){

    int err;                            // error code returned from api calls

    size_t globalFoph1;             // global domain size for our calculation
    size_t globalFoph2;

    size_t localFoph1;                  // local domain size for our calculation
    size_t localFoph2;

    size_t vectorSize = sizeof(float) * dFeatures;


    cl_device_id device_id;             // compute device id
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel foph1;                   // compute kernel
    cl_kernel foph2;

    // Connect to a compute device
    //
    int gpu = 0;
    err = clGetDeviceIDs(NULL, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to create a device group!\n");
        return err;
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
    FILE *kernelFile = fopen($kernel_path, "rb");
    if (kernelFile == NULL) {
        printf("Error: Coudn't open kernel file.\n");
        return err;
    }

    fseek(kernelFile, 0 , SEEK_END);
    long kernelFileSize = ftell(kernelFile);
    rewind(kernelFile);

    // Allocate memory to hold kernel
    //
    char *KernelSource = malloc(kernelFileSize*sizeof(char));
    memset(KernelSource, 0, kernelFileSize);
    if (KernelSource == NULL) {
        printf("Error: failed to allocate memory to hold kernel text.\n");
        return err;
    }

    // Read the kernel into memory
    //
    int result = fread(KernelSource, sizeof(char), kernelFileSize, kernelFile);
    if (result != kernelFileSize) {
        printf("Error: read fewer bytes of kernel text than expected.\n");
        return err;
    }
    fclose(kernelFile);

    printf("%s\n", KernelSource);
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
        char buffer[32768];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
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
    // Create the compute kernel in the program we wish to run
    //
    foph2 = clCreateKernel(program, "firstOrderPhaseTwo", &err);
    if (!foph2 || err != CL_SUCCESS){
        printf("Error: Failed to create compute kernel!\n");
        return err;
    }
    // Set the local work group sizes for executing the kernel on the device
    //
    err = clGetKernelWorkGroupInfo(foph1, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localFoph1), &localFoph1, NULL);
    err |= clGetKernelWorkGroupInfo(foph2, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localFoph2), &localFoph2, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        return err;
    }
    // Set the global work group sizes
    // Set to the first multiple of local size greater than nPoints
    int num_foph1_workgroups = (nPoints % localFoph1 != 0)?(nPoints / localFoph1 + 1):(nPoints/localFoph1);
    globalFoph1 = num_foph1_workgroups * localFoph1;
    globalFoph2 = localFoph2;

    // convenient buffer receiving data from device
    float results[8];

    // Make Device Data
    //

    cl_mem d_results = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                    sizeof results, NULL, &err);

    cl_mem d_input_data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    vectorSize * nPoints, input_data, &err);

    cl_mem d_labels = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * nPoints, labels, &err);

    cl_mem d_trainingAlpha = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * nPoints, training_alpha, &err);

    float *F = (float *)malloc(nPoints * sizeof(float));
    float *kernelDiag = (float *)malloc(nPoints * sizeof(float));

    // intialize values
    for(int i = 0; i < nPoints; i++){
        float *vecA = input_data + i * dFeatures;
        kernelDiag[i] = ${kernelFunc}Self(vecA, dFeatures, paramA, paramB, paramC);
        F[i] = - labels[i];
    }

    cl_mem d_kernelDiag = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * nPoints, kernelDiag, &err);

    cl_mem d_F = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                    sizeof(float) * nPoints, F, &err);

    cl_mem d_LowFs = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * num_foph1_workgroups, NULL, &err);

    cl_mem d_HighFs = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * num_foph1_workgroups, NULL, &err);

    cl_mem d_LowIndices = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * num_foph1_workgroups, NULL, &err);

    cl_mem d_HighIndices = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    sizeof(float) * num_foph1_workgroups, NULL, &err);

    if (err != CL_SUCCESS){
        printf("Error: Failed to allocate device memory!\n");
        return err;
    }


    // start of actual program
    // initialize

    // find iHigh and iLow
    int iLow = -1;
    int iHigh = -1;
    for(int i = 0; i < nPoints; i++){
        int label = labels[i];
        if(iLow < 0 && label == -1){
            iLow = i;
        }else if(iHigh < 0 && label == 1){
            iHigh = i;
        }
    }

    float bLow = 1.0f;
    float bHigh = -1.0f;
    float gap = bHigh - bLow;

    // do the first half iteration
    float alpha2old = training_alpha[iLow];
    float alpha1old = training_alpha[iHigh];
    float alphadiff = alpha2old - alpha1old;
    float alpha1diff;
    float alpha2diff;
    float alpha1new;
    float alpha2new;
    float alphaSum;

    int lowLabel = labels[iLow];
    int sign = labels[iHigh]*lowLabel;
    float L;
    float H;

    // find lower and upper bounds L and H
    if (sign < 0){
        if(alphadiff < 0.0f){
            L = 0.0f;
            H = cost + alphadiff;
        }else{
            L = alphadiff;
            H = cost;
        }
    }else{
        alphaSum = alpha2old + alpha1old;
        if (alphaSum < cost){
            L = 0.0f;
            H = alphaSum;
        }else{
            L = cost - alphaSum;
            H = cost;
        }
    }

    // compute and clip alpha2new but only if eta is positive, i.e. second derivative is negative
    float eta = kernelDiag[iLow] + kernelDiag[iHigh];
    float *vecA = input_data + iHigh * dFeatures;
    float *vecB = input_data + iLow * dFeatures;
    float phiAB = ${kernelFunc}(vecA, vecB, dFeatures, paramA, paramB, paramC);
    eta -= 2.0f * phiAB;
    if (eta > 0.0f){
        //compute
        alpha2new = alpha2old + labels[iLow]*gap/eta;
        //clip
        if (alpha2new < L){
            alpha2new = L;
        }else if(alpha2new > H){
            alpha2new = H;
        }else{ // alpha2new can now only assume endpoints or alpha2old (this is rare)
            float slope = lowLabel * gap;
            float delta = slope * (H - L);
            if (delta > 0){
                if (slope > 0){
                    alpha2new = H;
                }else{
                    alpha2new = L;
                }
            }else{
                alpha2new = alpha2old;
            }
        }
    }

    alpha2diff = alpha2new - alpha2old;
    alpha1diff = -sign*alpha2diff;
    alpha1new = alpha1old + alpha1diff;
    training_alpha[iHigh] = alpha1new;
    training_alpha[iLow] = alpha2new;
    int iteration;
    for (iteration = 1; true; iteration++){
        if(bLow <= bHigh + 2 * tolerance){
            break;
        }
        if ((iteration & 0x7ff) == 0) {
           printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
        }

        if (heuristic == 0){
            // Set the arguments to foph1
            //
            err = 0;
            err  = clSetKernelArg(foph1, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(foph1, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(foph1, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(foph1, 3, sizeof(cl_mem), &d_F);
            err  |= clSetKernelArg(foph1, 4, sizeof(cl_mem), &d_LowFs);
            err  |= clSetKernelArg(foph1, 5, sizeof(cl_mem), &d_HighFs);
            err  |= clSetKernelArg(foph1, 6, sizeof(cl_mem), &d_LowIndices);
            err  |= clSetKernelArg(foph1, 7, sizeof(cl_mem), &d_HighIndices);
            err  |= clSetKernelArg(foph1, 8, sizeof(int), &nPoints);
            err  |= clSetKernelArg(foph1, 9, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(foph1, 10, sizeof(float), &epsilon);
            err  |= clSetKernelArg(foph1, 11, sizeof(float), &Ce);
            err  |= clSetKernelArg(foph1, 12, sizeof(int), &iHigh);
            err  |= clSetKernelArg(foph1, 13, sizeof(int), &iLow);
            err  |= clSetKernelArg(foph1, 14, sizeof(float), &alpha1diff);
            err  |= clSetKernelArg(foph1, 15, sizeof(float), &alpha2diff);
            err  |= clSetKernelArg(foph1, 16, sizeof(float), &paramA);
            err  |= clSetKernelArg(foph1, 17, sizeof(float), &paramB);
            err  |= clSetKernelArg(foph1, 18, sizeof(float), &paramC);
            err  |= clSetKernelArg(foph1, 19, sizeof(int) * localFoph1, NULL);
            err  |= clSetKernelArg(foph1, 20, sizeof(float) * localFoph1, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }
            err = clEnqueueNDRangeKernel(commands, foph1, 1, NULL, &globalFoph1, &localFoph1, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel!\n");
                return err;
            }

            // Set the arguments to foph2
            //
            err = 0;
            err  = clSetKernelArg(foph2, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(foph2, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(foph2, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(foph2, 3, sizeof(cl_mem), &d_kernelDiag);
            err  |= clSetKernelArg(foph2, 4, sizeof(cl_mem), &d_LowFs);
            err  |= clSetKernelArg(foph2, 5, sizeof(cl_mem), &d_HighFs);
            err  |= clSetKernelArg(foph2, 6, sizeof(cl_mem), &d_LowIndices);
            err  |= clSetKernelArg(foph2, 7, sizeof(cl_mem), &d_HighIndices);
            err  |= clSetKernelArg(foph2, 8, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(foph2, 9, sizeof(float), &cost);
            err  |= clSetKernelArg(foph2, 10, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(foph2, 11, sizeof(float), &paramA);
            err  |= clSetKernelArg(foph2, 12, sizeof(float), &paramB);
            err  |= clSetKernelArg(foph2, 13, sizeof(float), &paramC);
            err  |= clSetKernelArg(foph2, 14, sizeof(int), &num_foph1_workgroups);
            err  |= clSetKernelArg(foph2, 15, sizeof(int) * localFoph2, NULL);
            err  |= clSetKernelArg(foph2, 16, sizeof(float) * localFoph2, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }
            err = clEnqueueNDRangeKernel(commands, foph2, 1, NULL, &globalFoph2, &localFoph2, 0, NULL, NULL);
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

            err = clEnqueueReadBuffer(commands, d_results, CL_TRUE, 0, sizeof(results), results, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to read buffer\n");
                return err;
            }
            bLow = results[0];
            bHigh = results[1];
            iLow = results[2];
            iHigh = results[3];
            alpha1diff = results[4];
            alpha2diff = results[5];
        }
    }
    printf("INFO: %d iterations\n", iteration);
    printf("INFO: bLow: %f, bHigh %f\n", bLow, bHigh);

    err = clEnqueueReadBuffer(commands, d_trainingAlpha, CL_TRUE, 0, sizeof(float) * nPoints, training_alpha, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to read buffer\n");
        return err;
    }

    // clean up
    clReleaseMemObject(d_results);
    clReleaseMemObject(d_input_data);
    clReleaseMemObject(d_labels);
    clReleaseMemObject(d_trainingAlpha);
    clReleaseMemObject(d_kernelDiag);
    clReleaseMemObject(d_F);
    clReleaseMemObject(d_LowFs);
    clReleaseMemObject(d_HighFs);
    clReleaseMemObject(d_LowIndices);
    clReleaseMemObject(d_HighIndices);
    clReleaseProgram(program);
    clReleaseKernel(foph1);
    clReleaseKernel(foph2);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    // save results
    float rho = (bHigh + bLow)/2;
    int nSV = 0;
    for(int i = 0; i < nPoints; i++){
        if (training_alpha[i] > epsilon){
            nSV++;
        }
    }
    trainResult[0] = rho;
    trainResult[1] = (float)nSV;
    trainResult[2] = (float)iteration;
    int svOffset = 3;
    int signedAlphaOffset = svOffset + nSV * dFeatures;
    int index = 0;
    for(int i = 0; i < nPoints; i++){
        if(training_alpha[i] > epsilon){
            memcpy(&input_data[index * dFeatures], &trainResult[svOffset + index * dFeatures], vectorSize);
            memcpy(&training_alpha[index], &trainResult[signedAlphaOffset + index], sizeof(float));
            index ++;
        }
    }
    // Shutdown and cleanup
    //
    return 0;
}