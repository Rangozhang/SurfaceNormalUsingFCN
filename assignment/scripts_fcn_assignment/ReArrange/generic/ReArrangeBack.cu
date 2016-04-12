#include "utils.h"
#include "common.h"
#include "THCDeviceTensor.cuh"
#include "THCDeviceTensorUtils.cuh"
#include "THCDeviceUtils.cuh"



template <typename Dtype>
__global__ void ReArrangeBackForward(int threads, const Dtype* input_data, Dtype* output_data, int samplenum, int channels, int height, int width ) {

  // int i,h,w,c;
  int imgsize = height * width * channels; 
  
  CUDA_KERNEL_LOOP(index, threads) 
  {
    int i = index / imgsize;
    int c = (index / (height * width)) % channels; 
    int h = (index / width) % height; 
    int w = index % width; 

    const Dtype* now_input = input_data + i * imgsize + (h * width + w) * channels;
    output_data[i * imgsize + c * height * width + h * width + w] = now_input[c];
    
  }
}


template <typename Dtype>
__global__ void ReArrangeBackBackward(int threads, const Dtype* gradOutput_data, Dtype* gradInput_data, int samplenum, int channels, int height, int width ) {

  // int i,h,w,c;
  int imgsize = height * width * channels; 

  CUDA_KERNEL_LOOP(index, threads) 
  {
    int i = index / imgsize;
    int c = (index / (height * width)) % channels; 
    int h = (index / width) % height; 
    int w = index % width; 

    Dtype * now_gradInput = gradInput_data + i * imgsize + (h * width + w) * channels;
    now_gradInput[c] = gradOutput_data[i * imgsize + c * height * width + h * width + w];
  }
}



static int cunn_ReArrangeBack_updateOutput(lua_State *L)
{

  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");


  int samplenum = luaT_getfieldcheckint(L, 1, "n");
  int height = luaT_getfieldcheckint(L, 1, "h");
  int width = luaT_getfieldcheckint(L, 1, "w");
  int channels = input->size[1];  
  int imgsize = height * width * channels;

  THCudaTensor_resize4d(state, output, samplenum, channels, height, width);

  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);
  float* output_data = THCudaTensor_data(state, output);

  int count = samplenum * imgsize;
  ReArrangeBackForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data, output_data, samplenum, channels, height, width );

  THCudaTensor_free(state, input);


  return 1;
}

static int cunn_ReArrangeBack_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  int samplenum = luaT_getfieldcheckint(L, 1, "n");
  int height = luaT_getfieldcheckint(L, 1, "h");
  int width = luaT_getfieldcheckint(L, 1, "w");
  int channels = input->size[1];  
  int imgsize = height * width * channels;

  THCudaTensor_resize2d(state, gradInput, samplenum * height * width, channels);

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  float* gradOutput_data = THCudaTensor_data(state, gradOutput);
  float* gradInput_data = THCudaTensor_data(state, gradInput);

  int count = samplenum * imgsize;
  ReArrangeBackBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, gradOutput_data, gradInput_data, samplenum, channels, height, width );

  THCudaTensor_free(state, gradOutput);


  return 1;
}



static const struct luaL_Reg cunn_ReArrangeBack__ [] = {
  {"ReArrangeBack_updateOutput", cunn_ReArrangeBack_updateOutput},
  {"ReArrangeBack_updateGradInput", cunn_ReArrangeBack_updateGradInput},
  {NULL, NULL}
};

void cunn_ReArrangeBack_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_ReArrangeBack__, "nn");
  lua_pop(L,1);
}



