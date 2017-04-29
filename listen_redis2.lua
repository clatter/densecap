
require 'torch'
require 'nn'
require 'image'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'

-- networking and serialization
local cmsgpack = require 'cmsgpack'
local redis = require 'redis'
-- TODO: import cmd utils

-- begin denscap
local init_opts = {
  checkpoint = '/home/rogowski/densecap/data/models/densecap/densecap-pretrained-vgg16.t7',
  image_size = 720,
  rpn_nms_thresh = 0.7,
  final_nms_thresh = 0.3,
  num_proposals = 1000,
  input_image = file_save_path,
  input_dir = '',
  input_split = '',
  splits_json = 'info/densecap_splits.json',
  vg_img_root_dir = '',
  max_images = 100,
  output_dir = '',
  output_vis = 1,
  output_vis_dir = 'vis/data',
  gpu = 0,
  use_cudnn = 1
}

function run_image(model, img_path, opt, dtype)

  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  -- Run the model forward
  local boxes, scores, captions = model:forward_test(img_caffe:type(dtype))
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes)

  local out = {
    img = img,
    boxes = boxes_xywh,
    scores = scores,
    captions = captions,
  }
  return out
end

function result_to_json(result)
  local out = {}
  out.boxes = result.boxes:float():totable()
  out.scores = result.scores:float():view(-1):totable()
  out.captions = result.captions
  return out
end

-- Load the model, and cast to the right type
local dtype, use_cudnn = utils.setup_gpus(init_opts.gpu, init_opts.use_cudnn)
local checkpoint = torch.load(init_opts.checkpoint)
local model = checkpoint.model
model:convert(dtype, use_cudnn)
model:setTestArgs{
  rpn_nms_thresh = init_opts.rpn_nms_thresh,
  final_nms_thresh = init_opts.final_nms_thresh,
  num_proposals = init_opts.num_proposals,
}
model:evaluate()


local client = redis.connect('127.0.0.1', 6379)
while true do
  local ident_in = client:blpop('densecap_in', 0)
  local ident = ident_in[2]
  local info = client:get('analyze:' .. ident)
  if info == nil then
    print('NOT FOUND')
    break
  end
  local info_object = cmsgpack.unpack(info)
  local file_save_path = '/home/rogowski/densecap/tmp/' .. ident .. info_object['fext']
  print(file_save_path)
  local raw_img = client:get('raw_bin:' .. ident)
  local f = assert(io.open(file_save_path, 'w'))
  f:write(raw_img)
  f:close()

  local result = run_image(model, file_save_path, init_opts, dtype)  
  local result_json = result_to_json(result)
  local boxes_set_result = client:set('odensity:boxes:'..ident, 
    cmsgpack.pack(result_json.boxes))
  local scores_set_result = client:set('odensity:scores:'..ident, 
    cmsgpack.pack(result_json.scores))
  local captions_set_result = client:set('odensity:captions:'..ident, 
    cmsgpack.pack(result_json.captions))
  break
end

