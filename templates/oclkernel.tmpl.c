// Kernel Functions
float linearSelf(__global float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float value = vecA[i];
        accumulant += value * value;
    }
    return accumulant;
}
float linear(__global float *vecA, __global float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        accumulant += vecA[i] * vecB[i];
    }
    return accumulant;
}

float gaussianSelf(__global float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    return 1.0f;
}
float gaussian(__global float *vecA, __global float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float diff = vecA[i] - vecB[i];
        accumulant += diff * diff;
    }
    return exp(- paramA * accumulant);
}
float polynomialSelf(__global float *vecA, int dFeatures, float paramA, float paramB, float paramC){
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
float polynomial(__global float *vecA, __global float *vecB, int dFeatures, float paramA, float paramB, float paramC){
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
float sigmoidSelf(__global float *vecA, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        float value = vecA[i];
        accumulant += value * value;
    }
    accumulant = accumulant * paramA + paramB;
    return tanh(accumulant);
}
float sigmoid(__global float *vecA, __global float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float accumulant = 0.0f;
    for(i = 0; i < dFeatures; i++){
        accumulant += vecA[i] * vecB[i];
    }
    accumulant = accumulant * paramA + paramB;
    return tanh(accumulant);
}
// helper functions
void storeResults(__global float *results, float bLow, float bHigh,
                int iLow, int iHigh, float alpha1diff, float alpha2diff){
    results[0] = bLow;
    results[1] = bHigh;
    results[2] = (float)iLow;
    results[3] = (float)iHigh;
    results[4] = alpha1diff;
    results[5] = alpha2diff;


}

void argMin(float valueA, float indexA, int valueB, int indexB, __local float *p_value, __local int *p_index){
    if (valueA < valueB){
        *p_value = valueA;
        *p_index = indexA;
    }else{
        *p_value = valueB;
        *p_index = indexB;
    }
}

void argMax(float valueA, float indexA, int valueB, int indexB, __local float *p_value, __local int *p_index){
    if (valueA > valueB){
        *p_value = valueA;
        *p_index = indexA;
    }else{
        *p_value = valueB;
        *p_index = indexB;
    }
}

void maxOperator(float valueA, float valueB, __local float *p_value){
    if (valueA > valueB){
        *p_value = valueA;
    }else{
        *p_value = valueB;
    }
}

void argMinReduce(int lid, size_t lsize, __local float *values, __local int *indices){
    for(int offset = lsize/2; offset > 0; offset >>= 1){
        if(lid < offset){
            argMin(values[lid], indices[lid], values[lid + offset], indices[lid + offset], values + lid, indices + lid);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void argMaxReduce(int lid, size_t lsize, __local float *values, __local int *indices){
    for(int offset = lsize/2; offset > 0; offset >>= 1){
        if(lid < offset){
            argMax(values[lid], indices[lid], values[lid + offset], indices[lid + offset], values + lid, indices + lid);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void maxReduce(int lid, size_t lsize, __local float *values){
    for(int offset = lsize/2; offset > 0; offset >>= 1){
        if(lid < offset){
            maxOperator(values[lid], values[lid + offset], values + lid);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

void updateAlphas(__global float *input_data, __global int *labels,
                    __global float *training_alpha, __global float *kernelDiag,
                    float cost, int dFeatures,
                    int iLow, int iHigh,
                    float bLow, float bHigh,
                    float paramA, float paramB, float paramC,
                    __global float *results){
    float gap = bHigh - bLow;
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
    __global float *vecA = input_data + iHigh * dFeatures;
    __global float *vecB = input_data + iLow * dFeatures;
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
        }
    }
    else{ // alpha2new can now only assume endpoints or alpha2old (this is rare)
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
    alpha2diff = alpha2new - alpha2old;
    alpha1diff = -sign * alpha2diff;
    alpha1new = alpha1old + alpha1diff;

    training_alpha[iHigh] = alpha1new;
    training_alpha[iLow] = alpha2new;
    storeResults(results, bLow, bHigh, iLow, iHigh, alpha1diff, alpha2diff);
    printf("iLow:%d iHigh:%d\n", (int)results[2], (int)results[3]);
    printf("bLow:%.2f bHigh:%.2f\n", results[0], results[1]);
    printf("alpha1diff:%.2f alpha2diff:%.2f\n", results[4], results[5]);
}
__kernel void initializeArrays(__global float *input_data, __global int *labels,
                                __global float *F, __global float *kernelDiag,
                                const int nPoints, const int dFeatures,
                                const float paramA, const float paramB, const float paramC){
    int gid = get_global_id(0);
    if(gid < nPoints){
        __global float *vecA = input_data + gid *dFeatures;
        kernelDiag[gid] = ${kernelFunc}Self(vecA, dFeatures, paramA, paramB, paramC);
        F[gid] = - (float)labels[gid];
    }
}
// only one thread of do first step is run.
__kernel void doFirstStep(__global float *input_data, __global int *labels,
                            __global float *training_alpha, __global float *kernelDiag,
                            const float cost, const int nPoints, const int dFeatures,
                            const float paramA, const float paramB, const float paramC,
                            __global float *results){
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
    updateAlphas(input_data, labels, training_alpha, kernelDiag, cost, dFeatures,
                iLow, iHigh, bLow, bHigh, paramA, paramB, paramC, results);
}

// runs with global size = npoints + padding, local size = max local size
__kernel void firstOrderPhaseOne(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *F,
                                __global float *lowFs, __global float *highFs,
                                __global int *lowIndices, __global int *highIndices,
                                const int nPoints, const int dFeatures,
                                const float epsilon, const float Ce,
                                int iHigh, int iLow,
                                float alpha1diff, float alpha2diff,
                                const float paramA, const float paramB,  const float paramC,
                                __local int *tempLocalIndices,
                                __local float *tempLocalFs){

    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int grpid = get_group_id(0);
    int lsize = get_local_size(0);
    float alpha;
    float f;
    int label;
    bool reduceHigh = false;
    bool reduceLow = false;
    if (gid < nPoints){
        alpha = training_alpha[gid];
        f = F[gid];
        label = labels[gid];

        if(alpha > epsilon) {
            if(alpha < Ce){
                reduceHigh = true;
                reduceLow = true;
            }else{
                if(label > 0){
                    reduceLow = true;
                }else{
                    reduceHigh = true;
                }
            }
        }else {
            if (label > 0){
                reduceHigh = true;
            }else{
                reduceLow = true;
            }
        }
        __global float *vecA = input_data + iHigh * dFeatures;
        __global float *vecB = input_data + iLow * dFeatures;
        __global float *vecC = input_data + gid * dFeatures;
        f += labels[iHigh] * alpha1diff * ${kernelFunc}(vecA,vecC,dFeatures,paramA,paramB,paramC);
        f += labels[iLow] * alpha2diff * ${kernelFunc}(vecB,vecC,dFeatures,paramA,paramB,paramC);
        F[gid] = f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);



    // 1st reduce (parallel, over all workgroups):
    // multiple local tempLocalFs[], each with size lsize -> one global highFs[], with size num_foph1_workgroups
    // works because lsize is multiple of 2.
    if(reduceHigh){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = FLT_MAX; // !!!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    argMinReduce(lid, lsize, tempLocalFs, tempLocalIndices);

    //store workgroup result in global array
    if (lid == 0){
        highIndices[grpid] = tempLocalIndices[0];
        highFs[grpid] = tempLocalFs[0];
    }

    // 1st reduce (parallel, over all workgroups):
    // multiple local tempLocalFs[], each with size lsize -> one global lowFs[], with size num_foph1_workgroups
    // works because lsize is multiple of 2.
    if(reduceLow){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = -FLT_MAX; // !!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    argMaxReduce(lid, lsize, tempLocalFs, tempLocalIndices);

    //store workgroup result in global array
    if (lid == 0){
        lowIndices[grpid] = tempLocalIndices[0];
        lowFs[grpid] = tempLocalFs[0];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}
//runs with global size = local size  = max local size, 1 workgroup
__kernel void firstOrderPhaseTwo(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *kernelDiag,
                                __global float *lowFs, __global float *highFs,
                                __global int *lowIndices, __global int *highIndices, __global float *results,
                                const float cost, const int dFeatures, const int num_foph1_workgroups,
                                const float paramA, const float paramB, const float paramC,
                                __local int *tempIndices, __local float *tempFs){
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    // 2nd Reduce: highFs[] w/ size num_foph1_workgroups (large) -> tempFs[] w/ size lsize (small).
    // Load first chunk of highFs into local memory
    if(lid < num_foph1_workgroups){
        tempIndices[lid] = highIndices[lid];
        tempFs[lid] = highFs[lid];
    }else{
        tempFs[lid] = FLT_MAX; // !!!!!
    }
    // 'comb' through the the rest of the chunks
    if (num_foph1_workgroups > lsize ){
        for (int i = lid + lsize; i < num_foph1_workgroups; i += lsize){

            argMin(highFs[i], highIndices[i], tempFs[lid], tempIndices[lid], tempFs+lid, tempIndices+lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempFs[] -> bHigh, tempIndices -> iHigh

    argMinReduce(lid, lsize, tempFs, tempIndices);

    int iHigh;
    float bHigh;
    if (lid == 0){
        iHigh = tempIndices[0];
        bHigh = tempFs[0];
    }
    // 2nd Reduce: highFs[] w/ size num_foph1_workgroups (large) -> tempFs[] w/ size lsize (small).
    // Load first chunk of highFs into local memory
    if(lid < num_foph1_workgroups){
        tempIndices[lid] = lowIndices[lid];
        tempFs[lid] = lowFs[lid];
    }else{
        tempFs[lid] = -FLT_MAX; // !!!!!
    }
    // 'comb' through the the rest of the chunks
    if (num_foph1_workgroups > lsize ){
        for (int i = lid + lsize; i < num_foph1_workgroups; i += lsize){
            argMax(lowFs[i], lowIndices[i], tempFs[lid], tempIndices[lid], tempFs+lid, tempIndices+lid);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempFs[] -> bHigh, tempIndices -> iHigh

    argMaxReduce(lid, lsize, tempFs, tempIndices);

    // update alpha using single thread
    if (lid == 0){
        int iLow = tempIndices[0];
        float bLow = tempFs[0];
        updateAlphas(input_data, labels, training_alpha, kernelDiag, cost, dFeatures,
                iLow, iHigh, bLow, bHigh, paramA, paramB, paramC, results);
    }
}

__kernel void secondOrderPhaseOne(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *F,
                                __global float *highFs, __global int *highIndices,
                                const int nPoints, const int dFeatures,
                                const float epsilon, const float Ce,
                                int iHigh, int iLow,
                                float alpha1diff, float alpha2diff,
                                const float paramA, const float paramB,  const float paramC,
                                __local int *tempLocalIndices,
                                __local float *tempLocalFs){
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int grpid = get_group_id(0);
    int lsize = get_local_size(0);
    float alpha;
    float f;
    int label;
    bool reduceHigh = false;
    if (gid < nPoints){
        alpha = training_alpha[gid];
        f = F[gid];
        label = labels[gid];

        if((label < 0 && alpha > epsilon)||(label > 0 && alpha < Ce)) {
            reduceHigh = true;
        }
        __global float *vecA = input_data + iHigh * dFeatures;
        __global float *vecB = input_data + iLow * dFeatures;
        __global float *vecC = input_data + gid * dFeatures;
        f += labels[iHigh] * alpha1diff * ${kernelFunc}(vecA,vecC,dFeatures,paramA,paramB,paramC);
        f += labels[iLow] * alpha2diff * ${kernelFunc}(vecB,vecC,dFeatures,paramA,paramB,paramC);
        F[gid] = f;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 1st reduce (parallel, over all workgroups):
    // multiple local tempLocalFs[], each with size lsize -> one global highFs[], with size num_foph1_workgroups
    // works because lsize is multiple of 2.
    if(reduceHigh){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = FLT_MAX; // !!!!!!
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    argMinReduce(lid, lsize, tempLocalFs, tempLocalIndices);

    //store workgroup result in global array
    if (lid == 0){
        highIndices[grpid] = tempLocalIndices[0];
        highFs[grpid] = tempLocalFs[0];
    }
}

__kernel void secondOrderPhaseTwo(__global float *highFs, __global int *highIndices,
                                __global float *results,
                                const int num_soph1_workgroups,
                                __local int *tempIndices, __local float *tempFs){
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    // 2nd Reduce: highFs[] w/ size num_foph1_workgroups (large) -> tempFs[] w/ size lsize (small).
    // Load first chunk of highFs into local memory
    if(lid < num_soph1_workgroups){
        tempIndices[lid] = highIndices[lid];
        tempFs[lid] = highFs[lid];
    }else{
        tempFs[lid] = FLT_MAX; // !!!!!
    }
    // 'comb' through the the rest of the chunks
    if (num_soph1_workgroups > lsize){
        for (int i = lid + lsize; i < num_soph1_workgroups; i += lsize){
            argMin(highFs[i], highIndices[i], tempFs[lid], tempIndices[lid], tempFs+lid, tempIndices+lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempFs[] -> bHigh, tempIndices -> iHigh
    argMinReduce(lid, lsize, tempFs, tempIndices);
    if (lid == 0){
        int iHigh = tempIndices[0];
        float bHigh = tempFs[0];
        results[1]= bHigh;
        results[3] = (float)iHigh;
    }
}

__kernel void secondOrderPhaseThree(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *F, __global float *kernelDiag,
                                __global float *lowFs, __global int *lowIndices, __global float *deltaFs,
                                __global float *results,
                                const int nPoints, const int dFeatures,
                                const float epsilon, const float Ce,
                                const float paramA, const float paramB,  const float paramC,
                                __local int *tempLocalIndices,
                                __local float *tempLocalValues){
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int lsize = get_local_size(0);
    int grpid = get_group_id(0);

    float bHigh = results[1];
    int iHigh = (int)results[3];

    barrier(CLK_LOCAL_MEM_FENCE);

    float alpha;
    float f;
    float label;
    bool reduceDelta = false;
    bool reduceF = false;
    float deltaF;
    float beta;
    float eta;

    if(gid < nPoints){
        alpha = training_alpha[gid];
        label = labels[gid];
        if((label > 0 && alpha > epsilon) || (label < 0 && alpha < Ce)){
            reduceF = true;
            f = F[gid];
            beta = bHigh - f;
            if(beta < epsilon){
                reduceDelta = true;
            }
        }
    }

    if(reduceF){
        tempLocalValues[lid] = f;
    }else{
        tempLocalValues[lid] = -FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    maxReduce(lid, lsize, tempLocalValues);
    if(lid == 0){
        lowFs[grpid] = tempLocalValues[0];
    }

    if(reduceDelta){
        __global float *vecA = input_data + iHigh * dFeatures;
        __global float *vecB = input_data + gid * dFeatures;
        eta = kernelDiag[iHigh] + kernelDiag[gid] - 2 * ${kernelFunc}(vecA, vecB, dFeatures, paramA, paramB, paramC);
        if(eta <= 0) eta = epsilon;
        deltaF = beta * beta / eta;
        tempLocalValues[lid] = deltaF;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalValues[lid] = -FLT_MAX;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMaxReduce(lid, lsize, tempLocalValues, tempLocalIndices);
    if(lid ==0){
        deltaFs[grpid] = tempLocalValues[0];
        lowIndices[grpid] = tempLocalIndices[0];
    }
}

__kernel void secondOrderPhaseFour(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *kernelDiag,
                                __global float *lowFs, __global float *lowIndices,
                                __global float *deltaFs, __global float *results,
                                const float cost, const int dFeatures, const int num_soph3_workgroups,
                                const float paramA, const float paramB, const float paramC,
                                __local int *tempIndices, __local float *tempValues){

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    // start by doing a two stage reduction to find iLow
    // Load first chunk of highFs into local memory
    if(lid < num_soph3_workgroups){
        tempIndices[lid] = lowIndices[lid];
        tempValues[lid] = deltaFs[lid];
    }else{
        tempValues[lid] = -FLT_MAX; // !!!!!
    }
    // 'comb' through the the rest of the chunks
    if (num_soph3_workgroups > lsize){
        for (int i = lid + lsize; i < num_soph3_workgroups; i += lsize){
            argMax(deltaFs[i], lowIndices[i], tempValues[lid], tempIndices[lid], tempValues+lid, tempIndices+lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempValues[] -> bHigh, tempIndices -> iHigh
    argMaxReduce(lid, lsize, tempValues, tempIndices);
    int iLow;
    if(lid == 0){
        iLow = tempIndices[0];
    }

    if(lid < num_soph3_workgroups){
        tempValues[lid] = lowFs[lid];
    }else{
        tempValues[lid] = -FLT_MAX; // !!!!!
    }
    // 'comb' through the the rest of the chunks
    if (num_soph3_workgroups > lsize){
        for (int i = lid + lsize; i < num_soph3_workgroups; i += lsize){
            maxOperator(lowFs[i], tempValues[lid], tempValues + lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempValues[] -> bHigh, tempIndices -> iHigh
    maxReduce(lid, lsize, tempValues);
    if(lid == 0){
        float bLow = tempValues[0];
        float bHigh = results[1];
        int iHigh = (int)results[3];

        updateAlphas(input_data, labels, training_alpha, kernelDiag, cost, dFeatures,
                iLow, iHigh, bLow, bHigh, paramA, paramB, paramC, results);
    }

}