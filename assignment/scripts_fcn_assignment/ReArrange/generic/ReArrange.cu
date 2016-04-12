#include "utils.h"
#include "common.h"

template <typename Dtype>
__global__ void ReArrangeForward(int threads, const Dtype* input_data, Dtype* output_data, int samplenum, int channels, int height, int width ) {

  // int i,h,w,c;
  int imgsize = height * width * channels; 

  CUDA_KERNEL_LOOP(index, threads) 
  {
    int i = index / imgsize;
    int c = (index / (height * width)) % channels; 
    int h = (index / width) % height; 
    int w = index % width; 

    Dtype * now_output = output_data + i * imgsize + (h * width + w) * channels;
    now_output[c] = input_data[i * imgsize + c * height * width + h * width + w];
  }
}


template <typename Dtype>
__global__ void ReArrangeBackward(int threads, const Dtype* gradOutput_data, Dtype* gradInput_data, int samplenum, int channels, int height, int width ) {

  // int i,h,w,c;
  int imgsize = height * width * channels; 

  CUDA_KERNEL_LOOP(index, threads) 
  {
    int i = index / imgsize;
    int c = (index / (height * width)) % channels; 
    int h = (index / width) % height; 
    int w = index % width; 
    
    const Dtype * now_gradOutput = gradOutput_data + i * imgsize + (h * width + w) * channels;
    gradInput_data[i * imgsize + c * height * width + h * width + w]= now_gradOutput[c];
  }

}


static int cunn_ReArrange_updateOutput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  //THTensor_(resizeAs)(output, input);
  int samplenum = input->size[0];
  int channels = input->size[1];
  int height = input->size[2];
  int width  = input->size[3]; 
  int imgsize = height * width * channels; 

  THCudaTensor_resize2d(state, output, samplenum * height * width, channels);

  input = THCudaTensor_newContiguous(state, input);
  float* input_data = THCudaTensor_data(state, input);
  float* output_data = THCudaTensor_data(state, output);
  int count = samplenum * imgsize;

  ReArrangeForward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, input_data, output_data, samplenum, channels, height, width );

  THCudaTensor_free(state, input);

  return 1;
}

static int cunn_ReArrange_updateGradInput(lua_State *L)
{
  THCState *state = getCutorchState(L);
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  
  long samplenum = input->size[0];
  long channels = input->size[1];
  long height = input->size[2];
  long width  = input->size[3]; 
  long imgsize = height * width * channels; 
 

  THCudaTensor_resize4d(state, gradInput, samplenum, channels, height, width);

  gradOutput = THCudaTensor_newContiguous(state, gradOutput);
  float* gradOutput_data = THCudaTensor_data(state, gradOutput);
  float* gradInput_data = THCudaTensor_data(state, gradInput);
  int count = samplenum * imgsize;
  
  ReArrangeBackward <<< GET_BLOCKS(count), CUDA_NUM_THREADS, 0, THCState_getCurrentStream(state) >>>
      (count, gradOutput_data, gradInput_data, samplenum, channels, height, width );

  THCudaTensor_free(state, gradOutput);

  return 1;
}



static const struct luaL_Reg cunn_ReArrange__ [] = {
  {"ReArrange_updateOutput", cunn_ReArrange_updateOutput},
  {"ReArrange_updateGradInput", cunn_ReArrange_updateGradInput},
  {NULL, NULL}
};

void cunn_ReArrange_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_ReArrange__, "nn");
  lua_pop(L,1);
}

