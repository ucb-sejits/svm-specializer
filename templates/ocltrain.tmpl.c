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

void getResults(cl_command_queue queue, cl_mem d_results, float *results, float* bLow, float *bHigh,
                int *iLow, int *iHigh, float *alpha1diff, float *alpha2diff);

int train(float *input_data, int *labels,
            float epsilon, float Ce, float cost, float tolerance,
            int heuristic, int nPoints,  int dFeatures,
            float paramA, float paramB, float paramC,
            size_t localInitSize, size_t globalInitSize,
            int num_foph1_workgroups, size_t localFoph1Size, size_t globalFoph1Size,
            size_t localFoph2Size, size_t globalFoph2Size,
            cl_mem d_input_data, cl_mem d_labels, cl_mem d_trainingAlpha, cl_mem d_kernelDiag, cl_mem d_F,
            cl_mem d_lowFs, cl_mem d_highFs, cl_mem d_lowIndices, cl_mem d_highIndices, cl_mem d_results,
            cl_command_queue queue, cl_kernel init, cl_kernel step1, cl_kernel foph1, cl_kernel foph2, float *p_rho, int *p_nSV, int *p_iterations,
            float **p_signedAlpha, float **p_supportVectors){


    int err;
    err = 0;
    err |= clSetKernelArg(init, 0, sizeof(cl_mem), &d_input_data);
    err |= clSetKernelArg(init, 1, sizeof(cl_mem), &d_labels);
    err |= clSetKernelArg(init, 2, sizeof(cl_mem), &d_F);
    err |= clSetKernelArg(init, 3, sizeof(cl_mem), &d_kernelDiag);
    err |= clSetKernelArg(init, 4, sizeof(int), &nPoints);
    err |= clSetKernelArg(init, 5, sizeof(int), &dFeatures);
    err |= clSetKernelArg(init, 6, sizeof(float), &paramA);
    err |= clSetKernelArg(init, 7, sizeof(float), &paramB);
    err |= clSetKernelArg(init, 8, sizeof(float), &paramC);
    if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
    }
    err = clEnqueueNDRangeKernel(queue, init, 1, NULL, &globalInitSize, &localInitSize, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to execute kernel init!\n");
        printf("Global Size:%zu, Local Size:%zu\n", globalInitSize, localInitSize);

        return err;
    }

    float bLow;
    float bHigh;
    int iLow;
    int iHigh;
    float alpha1diff;
    float alpha2diff;
    size_t globalStep1Size = 1;
    size_t localStep1Size = 1;
    err = 0;
    err |= clSetKernelArg(step1, 0, sizeof(cl_mem), &d_input_data);
    err |= clSetKernelArg(step1, 1, sizeof(cl_mem), &d_labels);
    err |= clSetKernelArg(step1, 2, sizeof(cl_mem), &d_trainingAlpha);
    err |= clSetKernelArg(step1, 3, sizeof(cl_mem), &d_kernelDiag);
    err |= clSetKernelArg(step1, 4, sizeof(float), &cost);
    err |= clSetKernelArg(step1, 5, sizeof(int), &nPoints);
    err |= clSetKernelArg(step1, 6, sizeof(int), &dFeatures);
    err |= clSetKernelArg(step1, 7, sizeof(float), &paramA);
    err |= clSetKernelArg(step1, 8, sizeof(float), &paramB);
    err |= clSetKernelArg(step1, 9, sizeof(float), &paramC);
    err |= clSetKernelArg(step1, 10, sizeof(cl_mem), &d_results);

    if (err != CL_SUCCESS){
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }
    err = clEnqueueNDRangeKernel(queue, step1, 1, NULL, &globalStep1Size, &localStep1Size, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to execute kernel step1!\n");
        printf("Global Size:%zu, Local Size:%zu", globalStep1Size, localStep1Size);
        return err;
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS){
        printf("Error: waiting for queue to finish failed\n");
        return err;
    }

    float results[8];
    //printf("size of results: %d", sizeof(results));
    getResults(queue, d_results, results, &bLow, &bHigh, &iLow, &iHigh, &alpha1diff, &alpha2diff);

    int iteration;
    for (iteration = 0; true; iteration++){

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
            err  |= clSetKernelArg(foph1, 4, sizeof(cl_mem), &d_lowFs);
            err  |= clSetKernelArg(foph1, 5, sizeof(cl_mem), &d_highFs);
            err  |= clSetKernelArg(foph1, 6, sizeof(cl_mem), &d_lowIndices);
            err  |= clSetKernelArg(foph1, 7, sizeof(cl_mem), &d_highIndices);
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
            err  |= clSetKernelArg(foph1, 19, sizeof(int) * localFoph1Size, NULL);
            err  |= clSetKernelArg(foph1, 20, sizeof(float) * localFoph1Size, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, foph1, 1, NULL, &globalFoph1Size, &localFoph1Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Foph1!\n");
                return err;
            }

            // Set the arguments to foph2
            //
            err = 0;
            err  = clSetKernelArg(foph2, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(foph2, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(foph2, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(foph2, 3, sizeof(cl_mem), &d_kernelDiag);
            err  |= clSetKernelArg(foph2, 4, sizeof(cl_mem), &d_lowFs);
            err  |= clSetKernelArg(foph2, 5, sizeof(cl_mem), &d_highFs);
            err  |= clSetKernelArg(foph2, 6, sizeof(cl_mem), &d_lowIndices);
            err  |= clSetKernelArg(foph2, 7, sizeof(cl_mem), &d_highIndices);
            err  |= clSetKernelArg(foph2, 8, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(foph2, 9, sizeof(float), &cost);
            err  |= clSetKernelArg(foph2, 10, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(foph2, 11, sizeof(int), &num_foph1_workgroups);
            err  |= clSetKernelArg(foph2, 12, sizeof(float), &paramA);
            err  |= clSetKernelArg(foph2, 13, sizeof(float), &paramB);
            err  |= clSetKernelArg(foph2, 14, sizeof(float), &paramC);
            err  |= clSetKernelArg(foph2, 15, sizeof(int) * localFoph2Size, NULL);
            err  |= clSetKernelArg(foph2, 16, sizeof(float) * localFoph2Size, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, foph2, 1, NULL, &globalFoph2Size, &localFoph2Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Foph2!\n");
                return err;
            }

            // Wait for the command queue to get serviced before reading back results
            //
            err = clFinish(queue);

            if (err != CL_SUCCESS){
                printf("Error: waiting for queue to finish failed\n");
                return err;
            }

            getResults(queue, d_results, results, &bLow, &bHigh, &iLow, &iHigh, &alpha1diff, &alpha2diff);

        }else{

        }
    }
    printf("INFO: %d iterations\n", iteration);
    printf("INFO: bLow: %f, bHigh %f\n", bLow, bHigh);

    // get training alpha
    float *training_alpha = (float *)malloc(sizeof(float) * nPoints);

    err = clEnqueueReadBuffer(queue, d_trainingAlpha, CL_TRUE, 0, sizeof(float) * nPoints, training_alpha, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to read buffer\n");
        return err;
    }
    // save results
    *p_rho = (bHigh + bLow)/2;
    int nSV = 0;
    for(int i = 0; i < nPoints; i++){
        if (training_alpha[i] > epsilon){
            nSV++;
        }
    }
    int index = 0;
    *p_nSV = nSV;
    *p_supportVectors = (float *)malloc(sizeof(float) * nSV * dFeatures);
    *p_signedAlpha = (float *)malloc(sizeof(float) * nSV);
    for(int i = 0; i < nPoints; i++){
        if(training_alpha[i] > epsilon){
            (* p_signedAlpha)[index] = labels[i] * training_alpha[i];
            for(int j = 0; j < dFeatures; j++){
                (* p_supportVectors)[index*dFeatures + j] = input_data[i * dFeatures + j];
            }
            index ++;
        }
    }
    // Shutdown and cleanup
    //
    return 0;
}
void getResults(cl_command_queue queue, cl_mem d_results, float results[8], float* bLow, float *bHigh,
                int *iLow, int *iHigh, float *alpha1diff, float *alpha2diff){
    int err;
    err = clEnqueueReadBuffer(queue, d_results, CL_TRUE, 0, 8 * sizeof(float), results, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to read buffer\n");
    }
    *bLow = results[0];
    *bHigh = results[1];
    *iLow = (int)results[2];
    *iHigh = (int)results[3];
    *alpha1diff = results[4];
    *alpha2diff = results[5];
//    printf("Host: iLow:%d iHigh:%d\n", (int)results[2], (int)results[3]);
//    printf("Host: bLow:%.2f bHigh:%.2f\n", results[0], results[1]);
//    printf("Host: alpha1diff:%.2f alpha2diff:%.2f\n", results[4], results[5]);


}