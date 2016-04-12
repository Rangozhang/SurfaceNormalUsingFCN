require('torch')
require('nn')
require('cunn')
require('nngraph')


-- common obj name to be freed
local common = {'output', 'gradInput', 'gradOutputBuffer'}

-- temporary buffer name other than output/gradInput
local t = {
   -- convolution
   ['nn.SpatialConvolution'] = {'finput', 'fgradInput'},
   ['nn.SpatialConvolutionMM'] = {'finput', 'fgradInput'},

   -- pooling
   ['nn.SpatialMaxPooling'] = {'indices'},
   ['nn.TemporalMaxPooling'] = {'indices'},
   ['nn.VolumetricMaxPooling'] = {'indices'},
   ['nn.SpatialFractionalMaxPooling'] = {'indices'},

   -- regularizer
   ['nn.BatchNormalization'] = {'buffer', 'buffer2', 'centered', 'normalized'},
   ['nn.SpatialBatchNormalization'] = {'buffer', 'buffer2','centered', 'normalized'},
   ['nn.Dropout'] = {'noise'},
   ['nn.SpatialDropout'] = {'noise'},

   -- transfer
   ['nn.PReLU'] = {'gradWeightBuf', 'gradWeightBuf2'},
   ['nn.RReLU'] = {'noise'},
   ['nn.LogSigmoid'] = {'buffer'},

   -- etc
   ['nn.Mean'] = {'_gradInput'},
   ['nn.Normalize'] = {'_output', 'norm', 'normp'},
   ['nn.PairwiseDistance'] = {'diff'},
   ['nn.Reshape'] = {'_input', '_gradOutput'},

   -- fbcunn
   ['nn.AbstractParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
   ['nn.DataParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
   ['nn.ModelParallel'] = {'homeGradBuffers', 'input_gpu', 'gradOutput_gpu', 'gradInput_gpu'},
}


local function free_table_or_tensor(val, name, field)
   if type(val[name]) == 'table' then
      val[name] = {}
   elseif type(val[name]) == 'userdata' then
      val[name] = field.new()
   end
end


local function is_member(name, t)
   if t == nil then
      return false
   end

   for _, value in pairs(t) do
      if name == value then
         return true
      end
   end
   return false
end




-- Taken and modified from Soumith's imagenet-multiGPU.torch code
-- https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua
local function sanitize_fit(model)
   local list = model:listModules()
   for _,val in ipairs(list) do
      print(val);
      for name,field in pairs(val) do
         -- print('name: ' .. name);
         -- print('field: ' .. field);

         -- remove ffi obj
         if torch.type(field) == 'cdata' then
            val[name] = nil

         -- remove common obj
         elseif is_member(name, common) then
            free_table_or_tensor(val, name, field)

         -- remove specific obj
         elseif is_member(name, t[val.__typename]) then
            free_table_or_tensor(val, name, field)
         end
      end
   end
   return model
end

local function sanitize(net)

    local netsave = sanitize_fit(net):clone() 
    netsave = netsave:float()


    -- super hacky sanitize  
    local gt_inputs = torch.Tensor(1, 3, 32, 32):float()
    netsave:forward(gt_inputs)
    local modellist = netsave:listModules()
    local layernum = #modellist
    local networkoutput = modellist[layernum].output:clone()
    netsave:backward(gt_inputs, networkoutput)

    for k, l in ipairs(netsave.modules) do

        -- clean up buffers
        local m = netsave.modules[k]
        m.output = m.output.new()
        m.gradInput = m.gradInput.new()
        m.finput =  m.finput and m.finput.new() or nil
        m.fgradInput =  m.fgradInput and m.fgradInput.new() or nil
        m.buffer = nil
        m.buffer2 = nil
        m.centered = nil
        m.std = nil
        m.normalized = nil
        m.indices = nil
        if m.weight then 
            m.weight = m.weight:clone()
            m.gradWeight = m.gradWeight:clone()
            m.bias = m.bias:clone()
            m.gradBias = m.gradBias:clone()
        end
    end
    netsave.output =  netsave.output.new()
    netsave.gradInput =  netsave.gradInput.new()
    netsave:apply(function(m) if m.weight then m.gradWeight = nil; m.gradBias = nil; end end)
    collectgarbage()

    return netsave

end


return  sanitize


