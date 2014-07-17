#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif
#include <time.h>


void getResults(cl_command_queue queue, cl_mem d_results, float *results, float* bLow, float *bHigh,
                int *iLow, int *iHigh, float *alpha1diff, float *alpha2diff);

typedef struct CacheNode CacheNode;
struct CacheNode{
    CacheNode *prev;
    CacheNode *next;
    int cacheLineNum;
    int index;
};
CacheNode* createCacheNode(int cacheLineNum, int index){
    CacheNode* node = calloc(1,sizeof(CacheNode));
    node->cacheLineNum = cacheLineNum;
    node->index = index;
    return node;
}
typedef struct CacheList CacheList;
struct CacheList{
    CacheNode *first;
    CacheNode *last;
};
CacheList* createList(){
    CacheList* list = calloc(1,sizeof(CacheList));
    return list;
}
void addToEnd(CacheList* list, CacheNode* node){
    if(list->last!=NULL){
        node->prev = list->last;
        list->last->next = node;
        list->last = node;
    }
    else{
        list->first = node;
        list->last = node;
    }
}
void removeNode(CacheList* list, CacheNode* node){
// only element in list
    if(node->next==NULL && node->prev==NULL){
        list->first = NULL;
        list->last = NULL;
        node->prev=NULL;
        node->next=NULL;
        return;
    }
    //end of list
    if(node->next==NULL){
        list->last = node->prev;
        node->prev=NULL;
        return;
    }
    //beginning of list
    if(node->prev==NULL){
        list->first = node->next;
        node->next=NULL;
        return;
    }
    //middle of list
    node->prev->next=node->next;
    node->next->prev=node->prev;
    node->next = NULL;
    node->prev = NULL;
}
void moveToEnd(CacheList *list, CacheNode* node){
    removeNode(list, node);
    addToEnd(list, node);
}
typedef struct Cache Cache;
struct Cache{
    // doubly linked list representing the order of use of cache lines
    //least recently used -> most recently used
    CacheList *list;
    int nPoints;
    int maxSize;
    int occupancy;
    int hits;
    int misses;
    // node pointers representation of the indices represented in the cache
    // maps indices to cacheList nodes, which maps to location in cache
    CacheNode **cacheMap;
};
Cache* createCache(int nPoints, int numLines){
    Cache* cache = calloc(1,sizeof(Cache));
    cache->hits = 0;
    cache->misses = 0;
    cache->nPoints = nPoints;
    cache->maxSize = numLines;
    cache->occupancy = 0;
    cache->list = createList();
    cache->cacheMap = calloc(nPoints,sizeof(CacheNode *));
    return cache;
}
void cacheCall(Cache *cache, int index, int *cacheIndex, bool *compute){
    CacheNode *requested = cache->cacheMap[index];
    CacheList *list = cache->list;
    if(cache->cacheMap[index] == NULL){// cache miss
        cache->misses++;
        *compute = true;
        if(cache->occupancy == cache->maxSize){
            //move first node to end
            requested = list->first;
            moveToEnd(list,requested);
            //rewrite map, overwriting the lru node and erasing it from the map
            cache->cacheMap[index] = requested;
            cache->cacheMap[requested->index] = NULL;
            int removed = requested->index;
            requested->index = index;
//            printf("Overwrote Node with index %d!\n",removed);
        }else{
            requested = createCacheNode(cache->occupancy,index);
            cache->cacheMap[index] = requested;
            addToEnd(list,requested);
            cache->occupancy++;
//            printf("Added node!\n");
        }
        *cacheIndex = requested->cacheLineNum;
    }else{// cache hit
        cache->hits++;
        //move to end
        moveToEnd(list, requested);
        *cacheIndex = requested->cacheLineNum;
        *compute = false;
    }
//    printf("Last node index: %d\n",cache->list->last->index);
}
void cacheTesting(){
    Cache* cache = createCache(10, 4);
    int cacheLineNum;
    bool compute;
    cacheCall(cache, 9, &cacheLineNum, &compute);
    cacheCall(cache, 0, &cacheLineNum, &compute);
    cacheCall(cache, 2, &cacheLineNum, &compute);
    cacheCall(cache, 2, &cacheLineNum, &compute);
    cacheCall(cache, 1, &cacheLineNum, &compute);
    cacheCall(cache, 9, &cacheLineNum, &compute);
    cacheCall(cache, 3, &cacheLineNum, &compute);
    printf("%d%d\n",cacheLineNum,compute);


}
int train(int cacheSizeInFloats, float *input_data, int *labels,
            float epsilon, float Ce, float cost, float tolerance,
            int heuristic, int nPoints,  int dFeatures,
            float paramA, float paramB, float paramC,
            size_t localInitSize, size_t globalInitSize,
            int numGroups_foph1, size_t localFoph1Size, size_t globalFoph1Size,
            size_t localFoph2Size, size_t globalFoph2Size,
            int numGroups_soph1, size_t localSoph1Size, size_t globalSoph1Size,
            size_t localSoph2Size, size_t globalSoph2Size,
            int numGroups_soph3, size_t localSoph3Size, size_t globalSoph3Size,
            size_t localSoph4Size, size_t globalSoph4Size,
            cl_mem d_input_data, cl_mem d_input_data_colmajor, cl_mem d_labels, cl_mem d_trainingAlpha, cl_mem d_kernelDiag, cl_mem d_F,
            cl_mem d_highFsFO, cl_mem d_highIndicesFO, cl_mem d_lowFsFO, cl_mem d_lowIndicesFO,
            cl_mem d_highFsSO1, cl_mem d_highIndicesSO1, cl_mem d_lowFsSO3, cl_mem d_lowIndicesSO3, cl_mem d_deltaFsSO3,
            cl_mem d_results, cl_mem d_cache,
            cl_command_queue queue, cl_kernel init, cl_kernel step1, cl_kernel foph1, cl_kernel foph2,
            cl_kernel soph1, cl_kernel soph2, cl_kernel soph3, cl_kernel soph4,
            float *p_rho, int *p_nSV, int *p_iterations,
            float **p_signedAlpha, float **p_supportVectors){
    printf("%d\n",cacheSizeInFloats);
    int numCacheLines = cacheSizeInFloats/nPoints;
    Cache *cache = createCache(nPoints,numCacheLines);
    printf("%d\n",cache->maxSize);
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
        return err;
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS){
        printf("Error: waiting for queue to finish failed\n");
        return err;
    }

    float results[8];
    getResults(queue, d_results, results, &bLow, &bHigh, &iLow, &iHigh, &alpha1diff, &alpha2diff);

    int iteration;
    bool iLowCompute;
    bool iHighCompute;
    int iLowCacheIndex;
    int iHighCacheIndex;
    clock_t startTot = clock(), diffTot, diffSoph1, diffSoph2, diffSoph3, diffSoph4, diffGetResults;

    for (iteration = 0; true; iteration++){

        if(bLow <= bHigh + 2 * tolerance){
            break;
        }
        if ((iteration & 0x7ff) == 0) {
            printf("iteration: %d; gap: %f\n",iteration, bLow - bHigh);
        }
        cacheCall(cache,iHigh, &iHighCacheIndex, &iHighCompute);
        cacheCall(cache,iLow, &iLowCacheIndex, &iLowCompute);
        if (heuristic == 0){
            // Set the arguments to foph1
            //

            err = 0;
            err  = clSetKernelArg(foph1, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(foph1, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(foph1, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(foph1, 3, sizeof(cl_mem), &d_F);
            err  |= clSetKernelArg(foph1, 4, sizeof(cl_mem), &d_lowFsFO);
            err  |= clSetKernelArg(foph1, 5, sizeof(cl_mem), &d_highFsFO);
            err  |= clSetKernelArg(foph1, 6, sizeof(cl_mem), &d_lowIndicesFO);
            err  |= clSetKernelArg(foph1, 7, sizeof(cl_mem), &d_highIndicesFO);
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
            err  |= clSetKernelArg(foph2, 4, sizeof(cl_mem), &d_lowFsFO);
            err  |= clSetKernelArg(foph2, 5, sizeof(cl_mem), &d_highFsFO);
            err  |= clSetKernelArg(foph2, 6, sizeof(cl_mem), &d_lowIndicesFO);
            err  |= clSetKernelArg(foph2, 7, sizeof(cl_mem), &d_highIndicesFO);
            err  |= clSetKernelArg(foph2, 8, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(foph2, 9, sizeof(float), &cost);
            err  |= clSetKernelArg(foph2, 10, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(foph2, 11, sizeof(int), &numGroups_foph1);
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
            // Set the arguments to soph1
            //

            clock_t startSoph1 = clock();
            alpha1diff = labels[iHigh] * alpha1diff;
            alpha2diff = labels[iLow] * alpha2diff;
            err = 0;
            err  = clSetKernelArg(soph1, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(soph1, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(soph1, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(soph1, 3, sizeof(cl_mem), &d_F);
            err  |= clSetKernelArg(soph1, 4, sizeof(cl_mem), &d_highFsSO1);
            err  |= clSetKernelArg(soph1, 5, sizeof(cl_mem), &d_highIndicesSO1);
            err  |= clSetKernelArg(soph1, 6, sizeof(int), &nPoints);
            err  |= clSetKernelArg(soph1, 7, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(soph1, 8, sizeof(float), &epsilon);
            err  |= clSetKernelArg(soph1, 9, sizeof(float), &Ce);
            err  |= clSetKernelArg(soph1, 10, sizeof(int), &iHigh);
            err  |= clSetKernelArg(soph1, 11, sizeof(int), &iLow);
            err  |= clSetKernelArg(soph1, 12, sizeof(float), &alpha1diff);
            err  |= clSetKernelArg(soph1, 13, sizeof(float), &alpha2diff);
            err  |= clSetKernelArg(soph1, 14, sizeof(float), &paramA);
            err  |= clSetKernelArg(soph1, 15, sizeof(float), &paramB);
            err  |= clSetKernelArg(soph1, 16, sizeof(float), &paramC);
            err  |= clSetKernelArg(soph1, 17, sizeof(int) * localSoph1Size, NULL);
            err  |= clSetKernelArg(soph1, 18, sizeof(float) * localSoph1Size, NULL);
            err  |= clSetKernelArg(soph1, 19, sizeof(cl_mem), &d_cache);
            err  |= clSetKernelArg(soph1, 20, sizeof(bool), &iHighCompute);
            err  |= clSetKernelArg(soph1, 21, sizeof(bool), &iLowCompute);
            err  |= clSetKernelArg(soph1, 22, sizeof(int), &iHighCacheIndex);
            err  |= clSetKernelArg(soph1, 23, sizeof(int),&iLowCacheIndex);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, soph1, 1, NULL, &globalSoph1Size, &localSoph1Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Soph1!\n");
                return err;
            }
            clFinish(queue);
            diffSoph1 = clock() - startSoph1;

            clock_t startSoph2 = clock();

            // Set the arguments to soph2
            //
            err = 0;
            err  |= clSetKernelArg(soph2, 0, sizeof(cl_mem), &d_highFsSO1);
            err  |= clSetKernelArg(soph2, 1, sizeof(cl_mem), &d_highIndicesSO1);
            err  |= clSetKernelArg(soph2, 2, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(soph2, 3, sizeof(int), &numGroups_soph1);
            err  |= clSetKernelArg(soph2, 4, sizeof(int) * localSoph2Size, NULL);
            err  |= clSetKernelArg(soph2, 5, sizeof(float) * localSoph2Size, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, soph2, 1, NULL, &globalSoph2Size, &localSoph2Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Soph2!\n");
                return err;
            }
            clFinish(queue);
            diffSoph2 = clock() - startSoph2;

            // Set the arguments to soph3
            //
            clFinish(queue);
            getResults(queue, d_results, results, &bLow, &bHigh, &iLow, &iHigh, &alpha1diff, &alpha2diff);
            clock_t startSoph3 = clock();
            cacheCall(cache, iHigh, &iHighCacheIndex, &iHighCompute);

            err = 0;
            err  = clSetKernelArg(soph3, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(soph3, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(soph3, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(soph3, 3, sizeof(cl_mem), &d_F);
            err  |= clSetKernelArg(soph3, 4, sizeof(cl_mem), &d_kernelDiag);
            err  |= clSetKernelArg(soph3, 5, sizeof(cl_mem), &d_lowFsSO3);
            err  |= clSetKernelArg(soph3, 6, sizeof(cl_mem), &d_lowIndicesSO3);
            err  |= clSetKernelArg(soph3, 7, sizeof(cl_mem), &d_deltaFsSO3);
            err  |= clSetKernelArg(soph3, 8, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(soph3, 9, sizeof(int), &iHigh);
            err  |= clSetKernelArg(soph3, 10, sizeof(float), &bHigh);
            err  |= clSetKernelArg(soph3, 11, sizeof(int), &nPoints);
            err  |= clSetKernelArg(soph3, 12, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(soph3, 13, sizeof(float), &epsilon);
            err  |= clSetKernelArg(soph3, 14, sizeof(float), &Ce);
            err  |= clSetKernelArg(soph3, 15, sizeof(float), &paramA);
            err  |= clSetKernelArg(soph3, 16, sizeof(float), &paramB);
            err  |= clSetKernelArg(soph3, 17, sizeof(float), &paramC);
            err  |= clSetKernelArg(soph3, 18, sizeof(int) * localSoph3Size, NULL);
            err  |= clSetKernelArg(soph3, 19, sizeof(float) * localSoph3Size, NULL);
            err  |= clSetKernelArg(soph3, 20, sizeof(cl_mem), &d_cache);
            err  |= clSetKernelArg(soph3, 21, sizeof(bool), &iHighCompute);
            err  |= clSetKernelArg(soph3, 22, sizeof(int), &iHighCacheIndex);



            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, soph3, 1, NULL, &globalSoph3Size, &localSoph3Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Soph3!\n");
                return err;
            }
            clFinish(queue);
            diffSoph3 = clock() - startSoph3;

            // Set the arguments to soph4
            //
            clock_t startSoph4 = clock();

            err = 0;
            err  = clSetKernelArg(soph4, 0, sizeof(cl_mem), &d_input_data);
            err  |= clSetKernelArg(soph4, 1, sizeof(cl_mem), &d_labels);
            err  |= clSetKernelArg(soph4, 2, sizeof(cl_mem), &d_trainingAlpha);
            err  |= clSetKernelArg(soph4, 3, sizeof(cl_mem), &d_kernelDiag);
            err  |= clSetKernelArg(soph4, 4, sizeof(cl_mem), &d_lowFsSO3);
            err  |= clSetKernelArg(soph4, 5, sizeof(cl_mem), &d_lowIndicesSO3);
            err  |= clSetKernelArg(soph4, 6, sizeof(cl_mem), &d_deltaFsSO3);
            err  |= clSetKernelArg(soph4, 7, sizeof(cl_mem), &d_results);
            err  |= clSetKernelArg(soph4, 8, sizeof(float), &cost);
            err  |= clSetKernelArg(soph4, 9, sizeof(int), &dFeatures);
            err  |= clSetKernelArg(soph4, 10, sizeof(int), &numGroups_soph3);
            err  |= clSetKernelArg(soph4, 11, sizeof(float), &paramA);
            err  |= clSetKernelArg(soph4, 12, sizeof(float), &paramB);
            err  |= clSetKernelArg(soph4, 13, sizeof(float), &paramC);
            err  |= clSetKernelArg(soph4, 14, sizeof(cl_mem), &d_F);
            err  |= clSetKernelArg(soph4, 15, sizeof(int) * localSoph4Size, NULL);
            err  |= clSetKernelArg(soph4, 16, sizeof(float) * localSoph4Size, NULL);

            if (err != CL_SUCCESS){
                printf("Error: Failed to set kernel arguments! %d\n", err);
                return err;
            }

            err = clEnqueueNDRangeKernel(queue, soph4, 1, NULL, &globalSoph4Size, &localSoph4Size, 0, NULL, NULL);
            if (err != CL_SUCCESS){
                printf("Error: Failed to execute kernel Soph4!\n");
                return err;
            }
            clFinish(queue);
            diffSoph4 = clock() - startSoph4;

            // Wait for the command queue to get serviced before reading back results
            //
            err = clFinish(queue);

            if (err != CL_SUCCESS){
                printf("Error: waiting for queue to finish failed\n");
                return err;
            }
//            clock_t startGetResults = clock();

            getResults(queue, d_results, results, &bLow, &bHigh, &iLow, &iHigh, &alpha1diff, &alpha2diff);
//            clFinish(queue);
//            diffGetResults = clock() - startGetResults;
            //printf("iLow:%d, iHigh:%d\n", (int)results[2], (int)results[3]);
        }

    }
    printf("INFO: %d iterations\n", iteration);
    printf("INFO: bLow: %f, bHigh %f\n", bLow, bHigh);
    diffTot = clock() - startTot;
    float msecTot = diffTot * 1000.0 / CLOCKS_PER_SEC;
    printf("Total time taken %.5f milliseconds\n", msecTot);
    float msecSoph1 = diffSoph1 * 1000.0 / CLOCKS_PER_SEC;
    printf("Soph1 time taken %.5f milliseconds\n", msecSoph1);
    float msecSoph2 = diffSoph2 * 1000.0 / CLOCKS_PER_SEC;
    printf("Soph2 time taken %.5f milliseconds\n", msecSoph2);
    float msecSoph3 = diffSoph3 * 1000.0 / CLOCKS_PER_SEC;
    printf("Soph3 time taken %.5f milliseconds\n", msecSoph3);
    float msecSoph4 = diffSoph4 * 1000.0 / CLOCKS_PER_SEC;
    printf("Soph4 time taken %.5f milliseconds\n", msecSoph4);
//    float msecGetResults = diffGetResults * 1000.0 / CLOCKS_PER_SEC;
//    printf("GetResult time taken %.5f milliseconds\n", msecGetResults);
    // get training alpha
    float *trainingAlpha = (float *)malloc(sizeof(float) * nPoints);

    err = clEnqueueReadBuffer(queue, d_trainingAlpha, CL_TRUE, 0, sizeof(float) * nPoints, trainingAlpha, 0, NULL, NULL);
    if (err != CL_SUCCESS){
        printf("Error: Failed to read buffer\n");
        return err;
    }
    // save results
    *p_rho = (bHigh + bLow)/2;
    int nSV = 0;
    for(int i = 0; i < nPoints; i++){
        if (trainingAlpha[i] > epsilon){
            nSV++;
        }
    }
    int index = 0;
    *p_nSV = nSV;
    *p_supportVectors = (float *)malloc(sizeof(float) * nSV * dFeatures);
    *p_signedAlpha = (float *)malloc(sizeof(float) * nSV);
    for(int i = 0; i < nPoints; i++){
        if(trainingAlpha[i] > epsilon){
            (* p_signedAlpha)[index] = labels[i] * trainingAlpha[i];
            for(int j = 0; j < dFeatures; j++){
                (* p_supportVectors)[index*dFeatures + j] = input_data[i * dFeatures + j];
            }
            index ++;
        }
    }
    // Shutdown and cleanup
    //
    printf("%d Cache Hits!\n",cache->hits);
    printf("%d Cache Misses!\n",cache->misses);
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
 //   printf("iLow:%d, iHigh:%d\n", (int)results[2], (int)results[3]);
//    printf("bLow:%.9f, bHigh:%.9f\n", results[0], results[1]);
//    printf("alpha1diff:%.2f alpha2diff:%.2f\n", results[4], results[5]);


}