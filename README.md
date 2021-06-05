# FFmpeg-trt-backend-test

This repo contains Python scripts which can generate TensorRT engines easily.The goal is to provide an easy way to test the TensorRT backend of the dnn_processing filter in FFmpeg. To generate TRT engines for testing, please install [TensorRT](https://developer.nvidia.com/tensorrt) first. Or more conveniently, use the [NGC TensorRT container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt).

After installing TensorRT, run:

`python3 generate_engine.py`

Two TRT engines will be generated, one with one channel and static shape (implicit batchsize) and one with 3 channels and dynamic shape (explicit batchsize).

To test the TensorRT backend, first config and build ffmpeg as:

`./configure --enable-cuda-nvcc --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --enable-libtensorrt --extra-ldflags=-lstdc++ --enable-shared`

`make && make install`

Then run:

`ffmpeg -i input.mp4 -vf scale=1280:720,format=rgb24,dnn_processing=dnn_backend=tensorrt:model=dynamic_c3f.trt:input=:output= output.mp4`

Note that the TensorRT backend only takes models with one input and one output, so you can leave the input and output name blank.