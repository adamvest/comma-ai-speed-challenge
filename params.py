import os


class Parameters():
	def __init__(self):
		self.n_processors = 8

		#frame setup
		self.seq_len = 5
		self.overlap = 4
		self.check_interval = 1
		self.use_optical_flow = False
		self.use_both = True

		if self.use_both:
			self.num_channels = 10
		elif self.use_optical_flow:
			self.num_channels = 4
		else:
			self.num_channels = 6

		#preprocessing
		self.resize_mode = "crop"  # choice: "crop" "rescale" None
		self.img_w = 640   # 608
		self.img_h = 380   # 184
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

		self.train_data_info_path = "./datainfo/train_df_%d_%d_%s.pickle" % (self.seq_len, self.overlap, input)
		self.valid_data_info_path = "./datainfo/valid_df_%d_%d_%s.pickle" % (self.seq_len, self.overlap, input)

		#model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0.5
		self.clip = None
		self.batch_norm = True

		#training
		self.batch_updates = True
		self.epochs = 20
		self.batch_size = 4
		self.pin_mem = True
		self.optim = {"opt": "Adagrad", "lr": 0.0005} # {"opt": "Adam"}

		# Pretrain, Resume training
		self.resume = False
		self.load_base_deepvo = False
		self.pretrained_flownet = None
		self.resume_t_or_v = ".train"
		self.load_model_path = "./models/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model%s" % self.resume_t_or_v

		base = "scratch"

		if self.resume:
			if self.load_base_deepvo:
				base = "base"
			else:
				base = "speeds"

		self.record_path = "./logs/speeds_finetuned_from_train.txt"
		self.save_model_path = "./models/speeds_from_%s_%d_%d_%s.model" % (base, self.seq_len, self.overlap, input)


		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
		if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
			os.makedirs(os.path.dirname(self.train_data_info_path))


par = Parameters()
