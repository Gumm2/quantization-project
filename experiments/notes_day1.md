1. What is quantization? 
Quantization is the process of mapping high-precision numerical values onto a smaller set of representable values, typically integers, in order to reduce memory usage and improve computational efficiency.
2. What is scale?
    In quantization, scale is mathematically defined as the largest absolute value in a tensor or dataset divided by the integer it is mapped to (typically the largest on the chosen scale). All remaining floating point values are divided by this scale and rounded to create a new set of integers that represent the "quantized" values.
3. What is MSE? 
    MSE is mean squared error, which in quantization represents the average square of the error between the original, unquantized values and the quantized rounded precision values.
4. What causes outlier problems?
    In quantization, outlier problems usually stem from an abnormally large magnitude of a floating point value from a tensor or dataset. That large value inherently creates a much larger scale, greatly increasing the integer jumps between unquantized values and the possibility to destroy data by having smaller values round to 0.
5. Why does INT4 perform worse than INT8?
    INT4 chooses a smaller range of scaling values (-7 to 7 or 8) as opposed to INT8's values of -127 to 128. Because of the smaller range, the scale becomes larger because the same numerical range must be represented using fewer integer levels. This leads to larger "jumps" in integers across converted floating points, leading to significantly higher MSE.
6. What surprised me today? 
    I was surprised by the fact that even across randomly generated tensors with thousands if not more data points, average MSE between INT4 and INT8 precision techniques held. I would have assumed that after running the experiment several times, the "random" value generator would have created enough outliers to override the discrepancy and create an insanely larger or smaller MSE ratio between the two, but that never happened.