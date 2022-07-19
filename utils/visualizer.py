from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch


class Visualizer(object):
	"""docstring for Visualizer"""
	def __init__(self, writer_name):
		super(Visualizer, self).__init__()
		self.writer_name = writer_name
		self.writer = SummaryWriter(comment=self.writer_name)
		
	def metric_visual(self, mode, model_id, scores):
		self.writer.add_scalars('eval/Cider(C)', {'mode {}'.format(mode): scores['C']}, model_id)
		self.writer.add_scalars('eval/Spice(S)', {'mode {}'.format(mode): scores['S']}, model_id)
		self.writer.add_scalars('eval/Meteor(M)', {'mode {}'.format(mode): scores['M']}, model_id)
		self.writer.add_scalars('eval/Rouge(R)', {'mode {}'.format(mode): scores['R']}, model_id)
		self.writer.add_scalars('eval/Bleu1(B1)', {'mode {}'.format(mode): scores['B1']}, model_id)
		self.writer.add_scalars('eval/Bleu2(B2)', {'mode {}'.format(mode): scores['B2']}, model_id)
		self.writer.add_scalars('eval/Bleu3(B3)', {'mode {}'.format(mode): scores['B3']}, model_id)
		self.writer.add_scalars('eval/Bleu4(B4)', {'mode {}'.format(mode): scores['B4']}, model_id)
	
	def loss_visual(self, loss, curr_iter):
		self.writer.add_scalars('train/Loss', {'total': loss}, curr_iter)

	def perplexity_visual(self, perplexity, curr_iter):
		self.writer.add_scalars('train/Perplexity', {'total': perplexity}, curr_iter)

	def codebook_visual(self, num_modes, samples_per_gpu, \
		mode_emb_unquantized, curr_iter, mode_encoder, device):
		# visualize
		idx = torch.tensor(range(num_modes)).to(device)
		codebook_np = mode_encoder.codebook.embedding(idx).cpu().detach().numpy()
		mode_emb_unquantized_np = mode_emb_unquantized.cpu().detach().squeeze().numpy()
		mat_samples = np.concatenate((codebook_np, mode_emb_unquantized_np), axis=0)
		mat_labels = []
		mat_labels.extend(["cb"]*num_modes)	      # label 1 refers to emb in codebook
		mat_labels.extend(["un"]*samples_per_gpu) # label 2 refers to unquantized emb
		self.writer.add_embedding(mat_samples, metadata=mat_labels, global_step=curr_iter)

	def mlc_visual(self, projected_img_mlc, projected_txt_mlc, curr_iter):
		# visualize
		projected_img_mlc = projected_img_mlc.cpu().detach().numpy()
		projected_txt_mlc = projected_txt_mlc.cpu().detach().numpy()

		mat_samples = np.concatenate((projected_img_mlc, projected_txt_mlc), axis=0)
		mat_labels = []
		for i in range(32):
			mat_labels.extend(["img{}".format(i)]*100)	      # label 1 refers to image concept
		for i in range(32, 64):
			mat_labels.extend(["txt{}".format(i-32)]*100) 		  # label 2 refers to text concept
		self.writer.add_embedding(mat_samples, metadata=mat_labels, global_step=curr_iter)

