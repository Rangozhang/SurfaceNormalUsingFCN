
local ReArrangeBack, Parent = torch.class('nn.ReArrangeBack', 'nn.Module')

function ReArrangeBack:__init(h, w, n)
   Parent.__init(self)
   self.h = h or 0
   self.w = w or 0 
   self.n = n or 1
end

function ReArrangeBack:updateOutput(input)
   
   local nowsize = input:size()
   assert(nowsize:size() == 2)
   assert(self.h > 0)
   assert(self.w > 0)
   local height = self.h
   local width = self.w
   local imgsize = height * width 
   local samplenum = math.floor( nowsize[1] / imgsize )
   local channels = nowsize[2]
   self.n = samplenum
   assert(height * width * samplenum == nowsize[1])

   input.nn.ReArrangeBack_updateOutput(self, input)

   return self.output

end

function ReArrangeBack:updateGradInput(input, gradOutput)

   local nowsize = input:size()
   assert(nowsize:size() == 2)
   assert(self.h > 0)
   assert(self.w > 0)
   local height = self.h
   local width = self.w
   local imgsize = height * width 
   local samplenum = math.floor( nowsize[1] / imgsize )
   local channels = nowsize[2]
   self.n = samplenum
   assert(height * width * samplenum == nowsize[1]) 

   input.nn.ReArrangeBack_updateGradInput(self, input, gradOutput)

   return self.gradInput
end