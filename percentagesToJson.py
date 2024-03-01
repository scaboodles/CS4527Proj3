#   og clip had 5780 frames over 5AM - 9PM => 16hrs
#   3:12 over 16hrs
#   57600 real secs compressed to 192 secs
#   300 realtime seconds / vid second
#   5 min realtime / vid second
#   cut 60 sec from start and 5 from end
#   300 mins realtime cut from start, 25 mins realtime cut from end
#   Trimmed time lapse: 10:00 AM - 8:35 PM

import json
from segment_anything import sam_model_registry, SamPredictor
from getPercentBlue import getBluePercent

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

frame_data = {}
predictor = SamPredictor(sam)
batches = 0
for i in range(3831): 
    frame_data[i] = getBluePercent(f'./frames/{"{:04d}".format(i)}.jpg', predictor)
    if(i % 100 == 0 and not i == 0):
        with open(f'./data/framedataBatch{batches}', 'w') as f:
            json.dump(frame_data, f, indent=4)
        batches += 1
        frame_data = {}

    print("frame analysis", i, "complete")

with open(f'./data/framedataBatch{batches}', 'w') as f:
    json.dump(frame_data, f, indent=4)