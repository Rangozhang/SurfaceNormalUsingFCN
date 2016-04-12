--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

local Threads = require 'threads'

local donkey_file = 'donkey.lua'

do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx -- torch.random(1,10000) 
            torch.manualSeed(seed)
            -- torch.setnumthreads(1)
            print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
            paths.dofile(donkey_file)
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile(donkey_file)
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

os.execute('mkdir -p '.. opt.save)


function merge_table(t1, t2)
   local t = {}
   for k,v in pairs(t2) do
      t[k] = v
   end
   for k,v in pairs(t1) do
      t[k] = v
   end
   return t
end
