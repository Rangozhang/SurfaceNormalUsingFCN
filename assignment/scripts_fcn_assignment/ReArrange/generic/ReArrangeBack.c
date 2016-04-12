#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ReArrangeBack.c"
#else


static int nn_(ReArrangeBack_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  long samplenum = luaT_getfieldcheckint(L, 1, "n");
  long height = luaT_getfieldcheckint(L, 1, "h");
  long width = luaT_getfieldcheckint(L, 1, "w");
  long channels = input->size[1];  
  long imgsize = height * width * channels;

  THTensor_(resize4d)(output, samplenum, channels, height, width);

  real* output_data = THTensor_(data)(output);
  real* input_data = THTensor_(data)(input);
  long i,h,w,c;
  #pragma omp parallel for private(h,w)
  for (i=0; i < samplenum; ++i)
  {
    for (h=0; h < height; ++h)
    {
    	for (w=0; w < width; ++w)
      {
        real * now_input = input_data + i * imgsize + (h * width + w) * channels;
        for (c = 0; c < channels; ++c)
        {
          output_data[i * imgsize + c * height * width + h * width + w] = now_input[c];
        }
      }
    }
  }

  return 1;
}

static int nn_(ReArrangeBack_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  
  long samplenum = luaT_getfieldcheckint(L, 1, "n");
  long height = luaT_getfieldcheckint(L, 1, "h");
  long width = luaT_getfieldcheckint(L, 1, "w");
  long channels = input->size[1];  
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
        real * now_gradInput = gradInput_data + i * imgsize + (h * width + w) * channels;
        for (c = 0; c < channels; ++c)
        {
          now_gradInput[c] = gradOutput_data[i * imgsize + c * height * width + h * width + w];
        }
      }
    }
  }

  return 1;
}



static const struct luaL_Reg nn_(ReArrangeBack__) [] = {
  {"ReArrangeBack_updateOutput", nn_(ReArrangeBack_updateOutput)},
  {"ReArrangeBack_updateGradInput", nn_(ReArrangeBack_updateGradInput)},
  {NULL, NULL}
};

static void nn_(ReArrangeBack_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(ReArrangeBack__), "nn");
  lua_pop(L,1);
}

#endif


