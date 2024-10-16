import numpy as np

def get_size_after_con2D(dim_in=(640, 2360), padding=[0, 0], dilation=[1, 1], kernel_size=[5, 5], stride=[1, 1]):
    if type(padding) == int:  padding = [padding, padding]
    if type(dilation) == int:  dilation = [dilation, dilation]
    if type(kernel_size) == int:  kernel_size = [kernel_size, kernel_size]
    if type(stride) == int:  stride = [stride, stride]

    if padding == 'same':
        assert stride in [1, [1, 1]], 'padding = same only valid for strides = 1'
        padding = [np.floor(kernel_size[0] / 2.0), np.floor(kernel_size[1] / 2.0)]

    dim0 = int((dim_in[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    dim1 = int((dim_in[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return dim0, dim1


def get_size_after_pooling2D(dim_in=(640, 2360), padding=[0, 0], dilation=[1, 1], kernel_size=[5, 5], stride=[1, 1]):
    if padding == 'same':
        assert stride in [1, [1, 1]], 'padding = same only valid for strides = 1'
        padding = [np.floor(kernel_size[0] / 2.0), np.floor(kernel_size[1] / 2.0)]

    if type(padding) == int:  padding = [padding, padding]
    if type(dilation) == int:  dilation = [dilation, dilation]
    if type(kernel_size) == int:  kernel_size = [kernel_size, kernel_size]
    if type(stride) == int:  stride = [stride, stride]

    dim0 = int((dim_in[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
    dim1 = int((dim_in[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)

    return dim0, dim1


def create_surrounding_mask(mask,surrounding_dist): #dim = batch, a,b
    # Create a mask with ones in the same shape as the input array
    try:
        sur_mask = np.zeros_like(mask)
        det_mask = mask
    except TypeError:
        sur_mask = np.zeros_like(mask.detach().cpu().numpy())
        det_mask = mask.detach().cpu().numpy()

    # Find the indices where the values are zero in the test array
    zero_indices = np.where(det_mask == 0)
    sur_mask[zero_indices]=1

    # Iterate over the zero indices
    for i in range(len(zero_indices[0])):
        row, col = zero_indices[0][i], zero_indices[1][i]
        # Set the surrounding entries within a distance of 5 pixels to one
        sur_mask[row, max(0, col-surrounding_dist):min(col+1+surrounding_dist, sur_mask.shape[1])] = 1

    return sur_mask
