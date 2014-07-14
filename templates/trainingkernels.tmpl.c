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
float sum(float4 in){
    return dot(in, (float4)(1.0f, 1.0f, 1.0f, 1.0f));
}
float gaussian(__global float *vecA, __global float *vecB, int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float4 accumulant = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float leftover = 0.0f;
    for(i = 0; i < dFeatures - 4; i+= 4){
        float4 diff = vload4(0,vecA + i) - vload4(0,vecB + i);
        accumulant += diff * diff;
    }
    for(; i < dFeatures; i++){
        float diff = vecA[i] - vecB[i];
        leftover += diff*diff;
    }
    leftover += sum(accumulant);
    return exp(paramA * leftover);
}
float gaussianDual(__global float *vecA, __global float *vecB, __global float *commonVec, float *resultA, float *resultB,
                int dFeatures, float paramA, float paramB, float paramC){
    int i;
    float4 accumulantA = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 accumulantB = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float A = 0.0f;
    float B = 0.0f;
    for(i = 0; i < dFeatures - 4; i+= 4){
        float4 val = vload4(0,commonVec + i);
        float4 diff = vload4(0,vecA + i) - val;
        accumulantA += diff * diff;
        diff = vload4(0,vecB + i) - val;
        accumulantB += diff * diff;
    }
    for(; i < dFeatures; i++){
        float val = commonVec[i];
        float diff = vecA[i] - val;
        A += diff*diff;
        diff = vecB[i] - val;
        B += diff*diff;
    }
    A += sum(accumulantA);
    B += sum(accumulantB);
    *resultA = exp(paramA * A);
    *resultB = exp(paramA * B);
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

void argMin(float valueA, int indexA, float valueB, int indexB, __local float *p_value, __local int *p_index){
    if (valueA < valueB){
        *p_value = valueA;
        *p_index = indexA;
    }else{
        *p_value = valueB;
        *p_index = indexB;
    }
}

void argMinPrivate(float valueA, int indexA, float valueB, int indexB, float *p_value, int *p_index){
    if (valueA < valueB){
        *p_value = valueA;
        *p_index = indexA;
    }else{
        *p_value = valueB;
        *p_index = indexB;
    }
}

void argMax(float valueA, int indexA, float valueB, int indexB, __local float *p_value, __local int *p_index){
    if (valueA > valueB){
        *p_value = valueA;
        *p_index = indexA;
    }else{
        *p_value = valueB;
        *p_index = indexB;
    }
}
void argMaxPrivate(float valueA, int indexA, float valueB, int indexB, float *p_value, int *p_index){
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
void maxOperatorPrivate(float valueA, float valueB, float *p_value){
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
__kernel void testBarrier(void){
    barrier(CLK_LOCAL_MEM_FENCE);
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
                    int iLow, int iHigh, float fLow,
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
        alpha2new = alpha2old + lowLabel*(bHigh - fLow)/eta;

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
                iLow, iHigh, bLow,bLow, bHigh, paramA, paramB, paramC, results);
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
        }else{
            if (label > 0){
                reduceHigh = true;
            }else{
                reduceLow = true;
            }
        }
        __global float *vecA = input_data + iHigh * dFeatures;
        __global float *vecB = input_data + iLow * dFeatures;
        __global float *vecC = input_data + gid * dFeatures;

        f +=  labels[iHigh] * alpha1diff * ${kernelFunc}(vecA,vecC,dFeatures,paramA,paramB,paramC);
        f += labels[iLow] *alpha2diff * ${kernelFunc}(vecB,vecC,dFeatures,paramA,paramB,paramC);
        F[gid] = f;
    }

    if(reduceHigh){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = FLT_MAX; // !!!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMinReduce(lid, lsize, tempLocalFs, tempLocalIndices);
    if (lid == 0){
        highIndices[grpid] = tempLocalIndices[0];
        highFs[grpid] = tempLocalFs[0];
    }

    if(reduceLow){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = -FLT_MAX; // !!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMaxReduce(lid, lsize, tempLocalFs, tempLocalIndices);
    if (lid == 0){
        lowIndices[grpid] = tempLocalIndices[0];
        lowFs[grpid] = tempLocalFs[0];
    }
}
//runs with global size = local size  = max local size, 1 workgroup
__kernel void firstOrderPhaseTwo(__global float *input_data, __global int *labels,
                                __global float *training_alpha, __global float *kernelDiag,
                                __global float *lowFs, __global float *highFs,
                                __global int *lowIndices, __global int *highIndices, __global float *results,
                                const float cost, const int dFeatures, const int numGroups_foph1,
                                const float paramA, const float paramB, const float paramC,
                                __local int *tempIndices, __local float *tempFs){
    int lid = get_local_id(0);
    int lsize = get_local_size(0);

    if(lid < numGroups_foph1){
        tempIndices[lid] = highIndices[lid];
        tempFs[lid] = highFs[lid];
    }else{
        tempFs[lid] = FLT_MAX; // !!!!!
    }
    if (numGroups_foph1 > lsize ){
        for (int i = lid + lsize; i < numGroups_foph1; i += lsize){

            argMin(highFs[i], highIndices[i], tempFs[lid], tempIndices[lid], tempFs+lid, tempIndices+lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMinReduce(lid, lsize, tempFs, tempIndices);
    int iHigh;
    float bHigh;
    if (lid == 0){
        iHigh = tempIndices[0];
        bHigh = tempFs[0];
    }

    if(lid < numGroups_foph1){
        tempIndices[lid] = lowIndices[lid];
        tempFs[lid] = lowFs[lid];
    }else{
        tempFs[lid] = -FLT_MAX; // !!!!!
    }
    if (numGroups_foph1 > lsize ){
        for (int i = lid + lsize; i < numGroups_foph1; i += lsize){
            argMax(lowFs[i], lowIndices[i], tempFs[lid], tempIndices[lid], tempFs+lid, tempIndices+lid);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMaxReduce(lid, lsize, tempFs, tempIndices);

    // update alpha using single thread
    if (lid == 0){
        int iLow = tempIndices[0];
        float bLow = tempFs[0];
        updateAlphas(input_data, labels, training_alpha, kernelDiag, cost, dFeatures,
                iLow, iHigh, bLow,bLow, bHigh, paramA, paramB, paramC, results);
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
                                __local float *tempLocalFs, __global float *iHighCache, bool iHighCompute){
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

        float kernelHigh;
        float kernelLow;
        __global float *vecB = input_data + iLow * dFeatures;
        __global float *vecC = input_data + gid * dFeatures;
        if (iHighCompute){
            //if(gid == 29) printf("cache miss\n");
            __global float *vecA = input_data + iHigh * dFeatures;
            ${kernelFunc}Dual(vecA, vecB, vecC, &kernelHigh, &kernelLow, dFeatures, paramA, paramB, paramC);
        }else {
            //if(gid == 29) printf("cache hit\n");


            kernelHigh = iHighCache[gid];
            kernelLow = ${kernelFunc}(vecB, vecC,dFeatures,paramA,paramB,paramC );
        }

        f += alpha1diff * kernelHigh;
        f +=  alpha2diff * kernelLow;
        F[gid] = f;
        //if(gid == 29) printf("%.4f iHighCache\n",iHighCache[gid]);
    }


    if(reduceHigh){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = FLT_MAX; // !!!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMinReduce(lid, lsize, tempLocalFs, tempLocalIndices);
    if (lid == 0){
        highIndices[grpid] = tempLocalIndices[0];
        highFs[grpid] = tempLocalFs[0];
    }
}

__kernel void secondOrderPhaseTwo(__global float *highFs, __global int *highIndices,
                                __global float *results,
                                const int numGroups_soph1,
                                __local int *tempIndices, __local float *tempFs){
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    float tempF;
    int tempI;
    if(lid < numGroups_soph1){
        tempI = highIndices[lid];
        tempF = highFs[lid];
    }else{
        tempFs[lid] = FLT_MAX; // !!!!!
    }
    if (numGroups_soph1 > lsize){
        for (int i = lid + lsize; i < numGroups_soph1; i += lsize){
            argMinPrivate(highFs[i], highIndices[i], tempF, tempI, &tempF, &tempI);
        }
        tempFs[lid] = tempF;
        tempIndices[lid] = tempI;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
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
                                const int iHigh, const float bHigh,
                                const int nPoints, const int dFeatures,
                                const float epsilon, const float Ce,
                                const float paramA, const float paramB,  const float paramC,
                                __local int *tempLocalIndices,
                                __local float *tempLocalValues,__global float *iHighCache){
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int lsize = get_local_size(0);
    int grpid = get_group_id(0);


    float alpha;
    float f;
    float label;
    bool reduceDelta = false;
    bool reduceF = false;
    float deltaF;
    float beta;
    float eta;
    float highKernel;
    if(gid < nPoints){

        alpha = training_alpha[gid];
        label = labels[gid];
        f = F[gid];
        beta = bHigh - f;
        if((label > 0 && alpha > epsilon) || (label < 0 && alpha < Ce)){
            reduceF = true;
            if(beta <= epsilon){
                reduceDelta = true;
            }
        }
        __global float *vecA = input_data + iHigh * dFeatures;
        __global float *vecB = input_data + gid * dFeatures;
        highKernel = ${kernelFunc}(vecA, vecB, dFeatures, paramA, paramB, paramC);
        iHighCache[gid] = highKernel;
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
    //if(gid == 29) printf("%.4f iHighCache\n",iHighCache[gid]);

    eta = kernelDiag[iHigh] + kernelDiag[gid] - 2 * highKernel;
    if(eta <= 0) eta = epsilon;
    deltaF = beta * beta / eta;
    if(reduceDelta){

        tempLocalValues[lid] = deltaF;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalValues[lid] = -FLT_MAX;
        tempLocalIndices[lid] = 0;
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
                                __global float *lowFs, __global int *lowIndices,
                                __global float *deltaFs, __global float *results,
                                const float cost, const int dFeatures, const int numGroups_soph3,
                                const float paramA, const float paramB, const float paramC, __global float* F,
                                __local int *tempIndices, __local float *tempValues){

    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    int tempI;
    float tempVal;
    if(lid < numGroups_soph3){
        tempI = lowIndices[lid];
        tempVal = deltaFs[lid];
    }else{
        tempValues[lid] = -FLT_MAX; // !!!!!
    }
    if (numGroups_soph3 > lsize){
        for (int i = lid + lsize; i < numGroups_soph3; i += lsize){
            argMaxPrivate(deltaFs[i], lowIndices[i], tempVal, tempI, &tempVal, &tempI);
        }
        tempValues[lid] = tempVal;
        tempIndices[lid] = tempI;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    argMaxReduce(lid, lsize, tempValues, tempIndices);
    int iLow;
    if(lid == 0){
        iLow = tempIndices[0];
    }

    if(lid < numGroups_soph3){
        tempVal = lowFs[lid];
    }else{
        tempValues[lid] = -FLT_MAX; // !!!!!
    }
    if (numGroups_soph3 > lsize){
        for (int i = lid + lsize; i < numGroups_soph3; i += lsize){
            maxOperatorPrivate(lowFs[i], tempVal, &tempVal);
        }
        tempValues[lid] = tempVal;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    maxReduce(lid, lsize, tempValues);

    if(lid == 0){
        float bLow = tempValues[0];

        float bHigh = results[1];
        int iHigh = (int)results[3];
        updateAlphas(input_data, labels, training_alpha, kernelDiag, cost, dFeatures,
                iLow, iHigh, F[iLow],bLow, bHigh, paramA, paramB, paramC, results);
    }

}