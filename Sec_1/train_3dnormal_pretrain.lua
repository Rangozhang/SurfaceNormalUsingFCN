require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require  'nnx'
require  'cunnx'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


local sanitize = require('sanitize')


----------------------------------------------------------------------
-- parse command-line options
-- TODO: put your path for saving models in "save" 
opt = lapp[[
  -s,--save          (default "/home/yu/seg_proj/FCN/Sec_1/")      subdirectory to save logs
  --saveFreq         (default 5)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -r,--learningRate  (default 0.01)      learning rate
  --learningRateDecay (default 0.5)    learning rate decay
  --learningRateDecay_every (default 2)   learning rate decay very number of epochs
  -b,--batchSize     (default 20)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 1)          gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --forceDonkeys     (default 0)
  --nDonkeys         (default 6)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.loadSize  = opt.scale 
-- TODO: setup the output size 
opt.labelSize = 16


opt.manualSeed = torch.random(1,10000) 
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end

opt.geometry = {3, opt.scale, opt.scale}
opt.outDim =  opt.classnum


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

if opt.network == '' then
  ---------------------------------------------------------------------
  -- TODO: load pretrain model and add some layers, let's name it as model_FCN for the sake of simplicity
  -- hint: you might need to add large padding in conv1 (perhaps around 100ish?) done
  -- hint2: use ReArrange instead of Reshape or View done
  -- hint3: you might need to skip the dropout and softmax layer in the model done
     model_FCN = torch.load('/home/yu/seg_proj/FCN/3d/models/AlexNet')
     model_FCN:remove(24)
     model_FCN:remove(23)
     model_FCN:remove(22)
     model_FCN:remove(19)
     model_FCN:remove(16)
     local tmp_mod = nn.SpatialConvolution(4096, 40, 1, 1, 1, 1)
     weights_init(tmp_mod)
     model_FCN:add(tmp_mod)
     model_FCN:add(nn.ReArrange())
     model_FCN:add(nn.LogSoftMax())
     model_FCN:get(1).padW = 100
     model_FCN:get(1).padH = 100
else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_FCN = tmp.FCN
end

-- print networks
print('fcn network:')
print(model_FCN)


-- TODO: loss function
criterion = nn.ClassNLLCriterion()

-- TODO: convert model and loss function to cuda
model_FCN:cuda()
criterion:cuda()

-- TODO: retrieve parameters and gradients
l1, l2 = model_FCN:parameters()
parameters, gradParameters = model_FCN:getParameters()

-- TODO: setup dataset, use data.lua
require 'data'

-- TODO: setup training functions, use fcn_train_cls.lua
fcn = require 'fcn_train_cls'

local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
   model_FCN:training()
   batchNumber = 0
   for i=1,opt.epochSize do
      donkeys:addjob(
         function()
            return makeData_cls_pre(trainLoader:sample(opt.batchSize))
         end,
         fcn.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()

end

epoch = 1
-- training loop
while epoch <= 10 do
  -- train/test
  train()

  if opt.learningRateDecay ~= 0 and epoch % opt.learningRateDecay_every == 0 then
    optimState.learningRate = optimState.learningRate * opt.learningRateDecay
    print('<trainer> learning rate decay to ' .. optimState.learningRate)
  end

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, 'checkpoint', string.format('fcn_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, { FCN = sanitize(model_FCN), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
