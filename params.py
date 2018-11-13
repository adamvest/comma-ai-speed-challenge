import os


class Parameters():
	def __init__(self):
		self.n_processors = 8

		#frame setup
		self.seq_len = 60
		self.overlap = 40
		self.check_interval = 1
		self.use_optical_flow = True
		self.use_both = False
		self.use_three_frames = True
		frames = 3 if self.use_three_frames else 2

		#preprocessing
		self.resize_mode = "rescale" # "crop", "rescale", or None
		self.img_w = 320  # 640
		self.img_h = 240  # 480
		self.img_means = [(-0.25843116, -0.21177726, -0.16270572),
			(0.24156884, 0.28822274, 0.33729428)]
		self.img_stds = (1, 1, 1)
		self.minus_point_5 = True

		#data info path
		input = "images"

		if self.use_both:
			input = "both"
		elif self.use_optical_flow:
			input = "flow"

		df_path = "./datainfo"
		self.train_data_info_path = os.path.join(df_path, "train_df_%d_%d_%d_%s.pickle" \
			% (self.seq_len, self.overlap, frames, input))
		self.valid_data_info_path = os.path.join(df_path, "valid_df_%d_%d_%d_%s.pickle" \
			% (self.seq_len, self.overlap, frames, input))
		self.test_data_info_path = os.path.join(df_path, "test_df_%d_%d_%d_%s.pickle" \
			% (self.seq_len, self.overlap, frames, input))

		#model
		self.model = "resnet3d" #deepvo
		self.resnet_depth = 34 # 18
		self.linear_size = 512 # 1024
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0.5
		self.clip = None
		self.batch_norm = True

		#number of channels
		mult = frames if self.model == "deepvo" else 1

		if self.use_both:
			self.num_channels = 5 * mult
		elif self.use_optical_flow:
			self.num_channels = 2 * mult
		else:
			self.num_channels = 3 * mult

		#training
		self.batch_updates = False
		self.epochs = 10
		self.batch_size = 2
		self.pin_mem = True
		self.optim = "Adam" # "Adagrad"

		#load weights
		model_dir = "./models"
		self.load_weights = False
		self.load_base_deepvo = False
		self.load_conv_only = False
		self.load_model_path = os.path.join(model_dir, "resnet3d_34_scratch_60_40_3_images.model.val_5")

		#testing
		self.evaluate_val = True

		#save paths
		base = "scratch"

		if self.load_weights:
			if self.load_base_deepvo:
				base = "base"
			else:
				base = "speeds"

		model = self.model if self.model == "deepvo" else self.model + "_%d" % self.resnet_depth

		self.save_model_path = os.path.join(model_dir, "%s_%s_%d_%d_%d_%s.model" \
			% (model, base, self.seq_len, self.overlap, frames, input))

		if not os.path.isdir(model_dir):
			os.makedirs(model_dir)
		if not os.path.isdir(df_path):
			os.makedirs(df_path)


par = Parameters()
