

// runs with global size = npoints + padding, local size = max local size
__kernel void firstOrderPhaseOne(__global float *input_data, int nPoints,int dFeatures){
    
    int gid = get_global_id(0);
    printf("d_input_data ,%d,: %.2f\n", gid,input_data[gid]);
}