require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

opt = lapp[[
  -n,--network       (default "")          reload pretrained network
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  --scale            (default 512)          scale of images to train on
  --classnum         (default 40) 
  --classification   (default 1)
  --usetri           (default 0)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
opt.network = '' -- your trained network
opt.list_file = '/scratch/16824/3d/list/testLabels.txt'
opt.path_dataset = '/scratch/16824/3d/croptest/'
opt.codebooktxt = '/scratch/16824/3d/list/codebook_40.txt' 
opt.resultpath =  '' -- save your results here 


print(opt)

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


model = torch.load(opt.network)
model_FCN = model.FCN:cuda()

collectgarbage()


opt.condDim = {3, opt.scale, opt.scale}

opt.div_num = 127.5
opt.datasize = 654


local list_file = opt.list_file
local path_dataset = opt.path_dataset

local f = assert(io.open(list_file, "r"))

function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.scale, opt.scale)
   input = input * 255
   return input
end



local codebooktxt = opt.codebooktxt
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 
  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()
end

function simplenorm(data)
  data:add(-127.5)
end



-- Get examples to plot
function getSamples( N, beg)

  local resultpath = opt.resultpath
  local N = N or 8
  local imgs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local namelist = {}

  for n = 1,N do
    if n + beg > opt.datasize then 
      break
    end
    filename = f:read("*line")
    table.insert(namelist, filename) 
    filename = path_dataset .. filename

    local sample = loadImage(filename)
    imgs[n] = sample:clone()
  end  
  simplenorm(imgs)
  -- TODO: if you are using pre-trained models, remember to convert RGB to BGR


  -- TODO: forward prop given imgs as inputs, output should be samples
  -- dimension of samples is batchsize * classnum(40) * height * width

  -- norms has the dimension of batchsize * 3 * height * width 
  -- following code convert the probabilities to normals using the codebook
  local norms = torch.Tensor((#samples)[1], 3, (#samples)[3], (#samples)[4])

  for i = 1, (#samples)[1] do 

    for c = 1, 3 do 
      codebook2 = torch.reshape(codebook[{{}, {c}}], opt.classnum ,1,1)
      codespatial = torch.expand(codebook2, opt.classnum,  (#samples)[3], (#samples)[4])
      nowoutput = torch.reshape( samples[{{i}, {}, {h}, {w}}], opt.classnum, (#samples)[3], (#samples)[4]) 
      results = torch.cmul(codespatial, nowoutput)
      results = torch.sum(results, 1):clone()
      norms[{{i},{c},{},{}}] = results:clone()

    end
  end

  -- normalize the outputs
  samples = norms:clone()
  sample_norm = torch.norm(samples, 2, 2)
  sample_norm = torch.cat({sample_norm, sample_norm, sample_norm}, 2)
  samples = torch.cdiv(samples, sample_norm)

  for i=1,N do

      if i + beg > opt.datasize then 
        break
      end
    
      output_name = paths.concat(resultpath, namelist[i])
      txt_name = paths.concat(resultpath, namelist[i]..'.txt')
      file = torch.DiskFile(txt_name, "w")
      nowsample = samples[i]:clone()

      nownorm = torch.totable(samples[i]:float())
      for c = 1, (#nowsample)[1] do 
        for w = 1, (#nowsample)[2] do 
          for h = 1, (#nowsample)[3] do 
            file:writeFloat( nownorm[c][h][w]) 
          end
        end
      end
      file:close()

      samples[i] = (samples[i] + 1 ) * opt.div_num
      output_img = samples[i]:clone()
      output_img = output_img:byte():clone()
      image.save(output_name, output_img )
  end

end


for i = 1,66 do 
  print(i)
  getSamples( 10, (i - 1) * 10 )
end

f:close()




