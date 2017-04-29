
local cmsgpack = require 'cmsgpack'
local redis = require 'redis'
-- TODO: import cmd utils
local client = redis.connect('127.0.0.1', 6379)

while true do
  local response = client:blpop('densecap_in', 0)
  print(response[2])
end
