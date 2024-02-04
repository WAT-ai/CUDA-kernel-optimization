extern "C" __global__
void conv2d(
    float *input,
    float *output,
    float *kernel,
    int input_width,
    int input_height,
    int kernel_width,
    int kernel_height
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Bounds checking
    if (row < input_height && col < input_width) {
        float sum = 0.0f;

        int start_row = row - kernel_height / 2;
        int start_col = col - kernel_width / 2;

        for (int i = 0; i < kernel_height; i++) {
            for (int j = 0; j < kernel_width; j++) {
                // Index in the input array
                int input_row = start_row + i;
                int input_col = start_col + j;

                if (input_row >= 0 && input_row < input_height && input_col >= 0 && input_col < input_width) {
                    sum += input[input_row * input_width + input_col] * kernel[i * kernel_width + j];
                }
            }
        }

        output[row * input_width + col] = sum;
    }
}