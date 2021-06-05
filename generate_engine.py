#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from functools import reduce
import numpy as np
import tensorrt
import pycuda.driver as cuda
import pycuda.autoinit
from trt_lite import TrtLite

np.set_printoptions(threshold=np.inf)

def build_engine_static(builder, input_shape):
    network = builder.create_network()
    data = network.add_input("data", tensorrt.DataType.FLOAT, input_shape)

    w = np.asarray(
        [0, 0, 0,
         0, 1, 0,
         0, 0, 0],
        dtype = np.float32)
    b = np.zeros((1,), np.float32)
    conv = network.add_convolution(data, 1, (3, 3), w, b)
    conv.stride = (1, 1)
    conv.padding = (1, 1)
    conv.get_output(0).name = "out"
    print('conv', conv.get_output(0).shape)

    network.mark_output(conv.get_output(0))

    builder.max_batch_size = 1
    builder.max_workspace_size = 1 << 30
    builder.fp16_mode = False

    return builder.build_cuda_engine(network)

def run_engine_static(save_and_load=False):
    batch_size = 1
    input_shape = (batch_size, 1, 720, 1280)
    output_shape = (batch_size, 1, 720, 1280)
    n = reduce(lambda x, y: x * y, input_shape)
    input_data = np.asarray(range(n), dtype=np.float32).reshape(input_shape)
    output_data = np.zeros(output_shape, dtype=np.float32)
    
    trt = TrtLite(build_engine_static, (input_shape[1:],))
    if save_and_load:
        trt.save_to_file("static_c1f_720*1280.trt")
        trt = TrtLite(engine_file_path="static_c1f_720*1280.trt")
    trt.print_info()

    d_buffers = trt.allocate_io_buffers(batch_size, True)

    cuda.memcpy_htod(d_buffers[0], input_data)
    trt.execute(d_buffers, batch_size)
    cuda.memcpy_dtoh(output_data, d_buffers[1])
    
    # print(output_data)

def build_engine_dynamic(builder):
    network = builder.create_network(1)
    data = network.add_input("data", tensorrt.DataType.FLOAT, (-1, 3, -1, -1))

    resize = network.add_resize(data)
    resize.scales = (1, 1, 2, 2)

    network.mark_output(resize.get_output(0))

    op = builder.create_optimization_profile()
    op.set_shape('data', (1, 3, 480, 640), (1, 3, 720, 1280), (16, 3, 1080, 1920))
    config = builder.create_builder_config()
    config.add_optimization_profile(op)
    
    config.flags = 1 << int(tensorrt.BuilderFlag.FP16)
    config.max_workspace_size = 1 << 30

    return builder.build_engine(network, config)

def run_engine_dynamic(save_and_load=False):
    input_shape = np.array([1, 3, 720, 1280])
    n = reduce(lambda x, y: x * y, input_shape)
    input_data = np.asarray(range(n), dtype=np.float32).reshape(input_shape)
    output_data = np.zeros(np.multiply(input_shape, (1,1,2,2)), dtype=np.float32)
    
    trt = TrtLite(build_engine_dynamic)
    if save_and_load:
        trt.save_to_file("dynamic_c3f.trt")
        trt = TrtLite(engine_file_path="dynamic_c3f.trt")
    trt.print_info()

    i2shape = {0: input_shape}
    d_buffers = trt.allocate_io_buffers(i2shape, True)

    cuda.memcpy_htod(d_buffers[0], input_data)
    trt.execute(d_buffers, i2shape)
    cuda.memcpy_dtoh(output_data, d_buffers[1])
    
    print(output_data)


if __name__ == '__main__':
    run_engine_static(True)
    run_engine_dynamic(True)