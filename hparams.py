# Audio:
num_mels = 80
num_freq = 1025
sample_rate = 20000
frame_length_ms = 50
frame_shift_ms = 12.5
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
griffin_lim_iters = 60
power = 1.5
signal_normalization = True
use_lws = False

# Text
text_cleaners = ['english_cleaners']

# Model
outputs_per_step = 3
max_iters = 1000 // outputs_per_step

# Train
batch_size = 32
# warm_up_epoch = 2
epochs = 10000
dataset_path = "dataset"
learning_rate = 1e-3
teacher_forcing_ratio = 1.0
# warm_up_learning_rate = 1e-4
# lr_drop = 0.5
# teacher_forced = 1.0
# teacher_forced_delay = 1.35e-5
# epochs_drop = 10.0
# weight_decay = 1e-6
checkpoint_path = "./model_new"
# warm_up_checkpoint_path = "./model_warm_up"
grad_clip_thresh = 1.0
decay_step = [20000, 50000, 100000]
save_step = 200
# warm_up_save_step = 200
log_step = 5
clear_Time = 20
