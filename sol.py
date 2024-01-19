def count_parameters_conv(in_channels: int, out_channels: int, kernel_size: int, bias: bool) -> int:
    params = in_channels * out_channels * kernel_size * kernel_size
    
    if bias:
        params += out_channels
    
    return params
