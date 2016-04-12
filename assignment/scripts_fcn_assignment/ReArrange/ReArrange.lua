
local ReArrange, Parent = torch.class('nn.ReArrange', 'nn.Module')

function ReArrange:__init()
   Parent.__init(self)
end

function ReArrange:updateOutput(input)
   local nowsize = input:size()
   assert(nowsize:size() == 4)
   input.nn.ReArrange_updateOutput(self, input)
   return self.output 
end

function ReArrange:updateGradInput(input, gradOutput)
   local nowsize = input:size()
   assert(nowsize:size() == 4)
   input.nn.ReArrange_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
