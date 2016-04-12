#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ReArrange.c"
#else


static int nn_(ReArrange_updateOutput)(lua_State *L)  // C macro here: nn_()
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor); //get input data from L, the 2nd argument, which is input
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor); //get property "output", from the 1st argument, which is self

  //THTensor_(resizeAs)(output, input);
  long samplenum = input->size[0];
  long channels = input->size[1];
  long height = input->size[2];
  long width  = input->size[3]; 
  long imgsize = height * width * channels; 
  THTensor_(resize2d)(output, samplenum * height * width, channels);

  real* output_data = THTensor_(data)(output);       //kind of get the changable pointer, like mutable_data in caffe
  real* input_data = THTensor_(data)(input);
  long i,h,w,c;
  #pragma omp parallel for private(h,w)
  for (i=0; i < samplenum; ++i)
  {
    for (h=0; h < height; ++h)
    {
    	for (w=0; w < width; ++w)
      {
        real * now_output = output_data + i * imgsize + (h * width + w) * channels;
        for (c = 0; c < channels; ++c)
        {
          now_output[c] = input_data[i * imgsize + c * height * width + h * width + w];
        }
      }
    }
  }

  return 1;
}

static int nn_(ReArrange_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  
  long samplenum = input->size[0];
  long channels = input->size[1];
  long height = input->size[2];
  long width  = input->size[3]; 
  long imgsize = height * width * channels; 

  THTensor_(resizeAs)(gradInput, input);

  const real* gradOutput_data = THTensor_(data)(gradOutput);
  real* gradInput_data = THTensor_(data)(gradInput);
  long i,h,w,c;
  #pragma omp parallel for private(h,w)
  for (i=0; i < samplenum; ++i)
  {
    for (h=0; h < height; ++h)
    {
      for (w=0; w < width; ++w)
      {
        const real * now_gradOutput = gradOutput_data + i * imgsize + (h * width + w) * channels;
        for (c = 0; c < channels; ++c)
        {
          gradInput_data[i * imgsize + c * height * width + h * width + w]= now_gradOutput[c];
        }
      }
    }
  }

  return 1;
}



static const struct luaL_Reg nn_(ReArrange__) [] = {
  {"ReArrange_updateOutput", nn_(ReArrange_updateOutput)},
  {"ReArrange_updateGradInput", nn_(ReArrange_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ReArrange_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ReArrange__), "nn");
  lua_pop(L,1);
}

#endif
