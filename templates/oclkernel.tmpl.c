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
void storeResults(__global float *results, float bLow, float bHigh,
                int iLow, int iHigh, float alpha1diff, float alpha2diff){
    results[0] = bLow;
    results[1] = bHigh;
    results[2] = (float)iLow;
    results[3] = (float)iHigh;
    results[4] = alpha1diff;
    results[5] = alpha2diff;


}
void argMax(int lid, size_t lsize, __local float *values, __local float *indices){
    for(int offset = lsize/2; offset > 0; offset >>= 1){
        if(lid < offset){
            float other = values[lid + offset];
            float mine = values[lid];
            float otherI = indices[lid + offset];
            float myI = indices[lid];
            if(mine > other){ // !!!!!
                values[offset] = mine;
                indices[offset] = myI;
            }else{
                values[offset] = other;
                indices[offset] = otherI;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
void argMin(int lid, size_t lsize, __local float *values, __local float *indices){
    for(int offset = lsize/2; offset > 0; offset >>= 1){
        if(lid < offset){
            float other = values[lid + offset];
            float mine = values[lid];
            float otherI = indices[lid + offset];
            float myI = indices[lid];
            if(mine < other){ // !!!!!
                values[offset] = mine;
                indices[offset] = myI;
            }else{
                values[offset] = other;
                indices[offset] = otherI;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
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
    __global float *vecA = input_data + iHigh * dFeatures;
    __global float *vecB = input_data + iLow * dFeatures;
    float phiAB = ${kernelFunc}(vecA, vecB, dFeatures, paramA, paramB, paramC);
//    printf("PhiAB: %.2f\n", phiAB);
    eta -= 2.0f * phiAB;
//    printf("Eta: %.2f\n", eta);
//    printf("gap: %.2f\n", gap);
//    printf("alpha1old: %.2f\n", alpha1old);
//    printf("alpha2old: %.2f\n", alpha2old);

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

//    printf("iLow:%d iHigh:%d\n", iLow, iHigh);
//    printf("bLow:%.2f bHigh:%.2f\n", bLow, bHigh);

    alpha2diff = alpha2new - alpha2old;
    alpha1diff = -sign * alpha2diff;
    alpha1new = alpha1old + alpha1diff;
//    printf("alpha1new: %.2f alpha2new: %.2f\n", alpha1new, alpha2new);

    training_alpha[iHigh] = alpha1new;
    training_alpha[iLow] = alpha2new;
    storeResults(results, bLow, bHigh, iLow, iHigh, alpha1diff, alpha2diff);

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
    // multiple local tempLocalFs[], each with size lsize -> one global lowFs[], with size num_foph1_workgroups
    // works because lsize is multiple of 2.
    if(reduceLow){
        tempLocalFs[lid] = f;
        tempLocalIndices[lid] = gid;
    }else{
        tempLocalFs[lid] = -FLT_MAX; // !!!!!
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    argMax(lid, lsize, tempLocalFs, tempLocalIndices);

    //store workgroup result in global array
    if (lid == 0){
        lowIndices[grpid] = tempLocalIndices[0];
        lowFs[grpid] = tempLocalFs[0];
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

    argMax(lid, lsize, tempLocalFs, tempLocalIndices);

    //store workgroup result in global array
    if (lid == 0){
        highIndices[grpid] = tempLocalIndices[0];
        highFs[grpid] = tempLocalFs[0];
    }
}
//runs with global size = local size  = max local size, 1 workgroup
void __kernel firstOrderPhaseTwo(__global float *input_data, __global int *labels,
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
            float otherf = highFs[i];
            float myf = tempFs[lid];
            int otheri = highIndices[i];
            int myi = tempIndices[lid];
            if(myf < otherf){ // !!!!!
                tempFs[lid] = myf;
                tempIndices[lid] = myi;
            }else{
                tempFs[lid] = otherf;
                tempIndices[lid] = otheri;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempFs[] -> bHigh, tempIndices -> iHigh

    argMin(lid, lsize, tempFs, tempIndices);

    int iHigh = tempIndices[0];
    float bHigh = tempFs[0];

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
            float otherf = lowFs[i];
            float myf = tempFs[lid];
            int otheri = lowIndices[i];
            int myi = tempIndices[lid];
            if(myf > otherf){ // !!!!!
                tempFs[lid] = myf;
                tempIndices[lid] = myi;
            }else{
                tempFs[lid] = otherf;
                tempIndices[lid] = otheri;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // 3rd Reduce (parallel, over single workgroup; this kernel should only be run with one workgroup):
    //tempFs[] -> bHigh, tempIndices -> iHigh

    argMax(lid, lsize, tempFs, tempIndices);

    int iLow = tempIndices[0];
    float bLow = tempFs[0];

    // update alpha using single thread
    if (lid == 0){
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
        __global float* vecA = input_data + iHigh * dFeatures;
        __global float* vecB = input_data + iLow * dFeatures;
        float phiAB = ${kernelFunc}(vecA, vecB, dFeatures, paramA, paramB, paramC);
        //printf("PhiAB: %.2f\n", phiAB);
        eta -= 2.0f * phiAB;
//        printf("Eta: %.2f\n", eta);
//        printf("gap: %.2f\n", gap);
//        printf("alpha1old: %.2f\n", alpha1old);
//        printf("alpha2old: %.2f\n", alpha2old);

        if (eta > 0.0f){
            //compute
            alpha2new = alpha2old + labels[iLow]*gap / eta;

            //clip
            if (alpha2new < L){
                alpha2new = L;
            }else if(alpha2new > H){
                alpha2new = H;
            }
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
//        printf("iLow:%d iHigh:%d\n", iLow, iHigh);
//        printf("bLow:%.2f bHigh:%.2f\n", bLow, bHigh);

        alpha2diff = alpha2new - alpha2old;
        alpha1diff = -sign * alpha2diff;
        alpha1new = alpha1old + alpha1diff;
        training_alpha[iHigh] = alpha1new;
        training_alpha[iLow] = alpha2new;
//        f("alpha1new: %.2f alpha2new: %.2f\n", alpha1new, alpha2new);
//        printf("iHigh: %d iLow: %d\n", iHigh, iLow);


        storeResults(results, bLow, bHigh, iLow, iHigh, alpha1diff, alpha2diff);
    }
}