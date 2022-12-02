import datetime
import os
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate
from . import Logger, visualize_ranking_grid
from .avgmeter import TorchTimeMeter, SingleMeter, \
    EpochMeter, EpochArrayMeter, LossEpochMetricsMeter
from .distribution import plot_body_parts_pairs_distance_distribution, plot_pairs_distance_distribution
from .engine_state import EngineStateListener
from .tools import perc
from .visualization.embeddings_projection import visualize_embeddings


class Writer(EngineStateListener):
    # TODO Integrate this with Pytorch Lightning
    """ A class to encapsulate external loggers and writers such as Tensorboard and Allegro ClearML
    """
    __main_writer = None  # type: Optional[Writer]

    @classmethod
    def current_writer(cls):
        # type: () -> Writer
        return cls.__main_writer

    def __init__(self, cfg):
        self.cfg = cfg
        self.model_name = cfg.project.start_time + cfg.project.experiment_id
        self.logger = Logger.current_logger()

        # running state
        self.is_training = True
        self.batch_debug_freq = cfg.train.batch_debug_freq

        # configs
        self.start_eval = cfg.test.start_eval
        self.eval_freq = cfg.train.eval_freq
        self.max_epoch = cfg.train.max_epoch

        # init time meters
        self.total_run_timer = TorchTimeMeter("total run time", False)
        self.test_timer = TorchTimeMeter("multi_target_test", False)
        self.epoch_timer = TorchTimeMeter("epoch", False)
        self.batch_timer = TorchTimeMeter("batch")
        self.data_loading_timer = TorchTimeMeter("data_loading", False)
        self.test_batch_timer = TorchTimeMeter("test_batch")
        self.performance_evaluation_timer = TorchTimeMeter("performance_evaluation", False)
        self.feature_extraction_timer = TorchTimeMeter("feature_extraction")
        self.loss_timer = TorchTimeMeter("loss_computation")
        self.optimizer_timer = TorchTimeMeter("optimizer_step")

        Writer.__main_writer = self

    def init_engine_state(self, engine_state, parts_num):
        self.engine_state = engine_state

        # init meters
        self.invalid_pairwise_distances_count = EpochMeter(self.engine_state)
        self.uncomparable_body_parts_pairs_count = EpochMeter(self.engine_state)
        self.invalid_pairs_count_at_test_time = SingleMeter(self.engine_state)
        self.uncomparable_queries_at_test_time = SingleMeter(self.engine_state)
        self.used_body_parts_in_max = EpochArrayMeter(self.engine_state, parts_num)
        self.losses = LossEpochMetricsMeter(engine_state)
        self.loss = EpochMeter(engine_state)

        # writer should be the last listener to be called
        self.engine_state.add_listener(self, True)

    ########################
    #    TRAINING STATS    #
    ########################

    def report_performance(self, cmc, mAP, ssmd, pxl_acc_avg, name=""):
        self.logger.add_scalar('r1 {}'.format(name), perc(cmc[0]), self.engine_state.epoch)
        self.logger.add_scalar('r5 {}'.format(name), perc(cmc[1]), self.engine_state.epoch)
        self.logger.add_scalar('r10 {}'.format(name), perc(cmc[2]), self.engine_state.epoch)
        self.logger.add_scalar('r20 {}'.format(name), perc(cmc[3]), self.engine_state.epoch)
        self.logger.add_scalar('mAP {}'.format(name), perc(mAP), self.engine_state.epoch)
        self.logger.add_scalar('ssmd {}'.format(name), ssmd, self.engine_state.epoch)
        self.logger.add_scalar('pxl_acc {}'.format(name), pxl_acc_avg, self.engine_state.epoch)

    def report_global_performance(self,
                                  cmc_per_dataset,
                                  mAP_per_dataset,
                                  ssmd_per_dataset,
                                  pxl_acc_per_dataset):
        self.logger.add_text('r1_global', str(cmc_per_dataset[0]))
        self.logger.add_text('r5_global', str(cmc_per_dataset[1]))
        self.logger.add_text('r10_global', str(cmc_per_dataset[2]))
        self.logger.add_text('r20_global', str(cmc_per_dataset[3]))
        self.logger.add_text('mAP_global', str(mAP_per_dataset))
        self.logger.add_text('ssmd_global', str(ssmd_per_dataset))
        self.logger.add_text('pxl_acc_global', str(pxl_acc_per_dataset))

    def intermediate_evaluate(self):
        return (self.engine_state.epoch + 1) >= self.start_eval and self.eval_freq > 0 and (
            self.engine_state.epoch + 1) % self.eval_freq == 0 and (self.engine_state.epoch + 1) != self.max_epoch

    def update_invalid_pairwise_distances_count(self, batch_pairwise_dist):
        self.invalid_pairwise_distances_count.update((batch_pairwise_dist == float(-1)).sum(), batch_pairwise_dist.nelement())

    def update_invalid_part_based_pairwise_distances_count(self, valid_body_part_pairwise_dist_mask):
        self.uncomparable_body_parts_pairs_count.update((valid_body_part_pairwise_dist_mask.nelement() - valid_body_part_pairwise_dist_mask.sum()),
                                                        valid_body_part_pairwise_dist_mask.nelement())

    def used_parts_statistics(self, M, body_part_id):
        # count apparition of each body part id, remove diagonal as we don't consider pairs with same id
        used_body_parts_count = torch.bincount(body_part_id.flatten(), minlength=M) - torch.bincount(
            body_part_id.diag(), minlength=M)
        used_body_parts_count = used_body_parts_count / 2  # body parts are counted two times since matrix is symmetric
        self.used_body_parts_in_max.update(used_body_parts_count, np.ones(len(used_body_parts_count))*used_body_parts_count.sum().item())

    # def plot_batch_distance_distribution(self, batch_pairwise_dist, labels):  # TODO report each epoch and not each batch for wandb performance issue
    #     if self.batch_debug_freq > 0 and (self.engine_state.global_step + 1) % self.batch_debug_freq == 0:
    #         batch_pairwise_dist = batch_pairwise_dist.detach().cpu().numpy()
    #         labels = labels.detach().cpu().numpy()
    #         if len(batch_pairwise_dist.shape) == 3:
    #             ssmd = plot_body_parts_pairs_distance_distribution(
    #                 batch_pairwise_dist, labels, labels, "Training batch")
    #             self.logger.add_scalar("SSMD/training batch ssmd", ssmd, self.engine_state.global_step)
    #         else:
    #             pos_p_mean, pos_p_std, neg_p_mean, neg_p_std, ssmd = plot_pairs_distance_distribution(
    #                 batch_pairwise_dist, labels, labels, "Training batch")
    #             self.logger.add_scalar("SSMD/training batch ssmd", ssmd, self.engine_state.global_step)

    # TODO batch plot
    # def report_body_parts_mean_distances(self, distances):  # TODO report each epoch and not each batch for wandb performance issue
    #     mean_distance_per_body_part = distances.mean(dim=(1, 2))
    #     for bp_id, count in enumerate(mean_distance_per_body_part):
    #         self.logger.add_scalar("Body parts mean distances/bp_{}".format(bp_id), count, self.engine_state.global_step)

    def visualize_triplets(self, images, masks, mask, dist):
        if self.batch_debug_freq > 0 and (self.engine_state.global_step + 1) % self.batch_debug_freq == 0:
            pass
            # TODO
            # np_mask = mask.clone().detach().cpu().numpy()
            # np_dist = dist.clone().detach().cpu().numpy()
            # dist_ap, dist_an = [], []
            # triplets = []
            # for i in range(20):
            #     print("Computing triplet {}".format(i))
            #     pos_d = np_dist[i]*(np_mask[i])
            #     neg_d = np_dist[i]*(np_mask[i] == 0)
            #     pos_idx = pos_d.argmax()
            #     neg_idx = neg_d.argmax()
            #
            #     # instance = (image, masks, id, body_part_id, body_part_name)
            #     pos_img = cv2.imread(images[pos_idx])
            #     pos_img = cv2.cvtColor(pos_img, cv2.COLOR_BGR2RGB)
            #     anc_img = cv2.imread(images[i])
            #     anc_img = cv2.cvtColor(anc_img, cv2.COLOR_BGR2RGB)
            #     neg_img = cv2.imread(images[neg_idx])
            #     neg_img = cv2.cvtColor(neg_img, cv2.COLOR_BGR2RGB)
            #
            #     pos = (pos_img, masks[pos_idx][bp], pos_idx, bp)
            #     anc = (anc_img, masks[i][bp], i, bp)
            #     neg = (neg_img, masks[neg_idx][bp], neg_idx, bp)
            #
            #     # pos, anc, neg, pos_dist, neg_dist = triplet
            #     triplet = [pos, anc, neg, pos_d[pos_idx], neg_d[neg_idx]]
            #
            #     triplets.append(triplet)
            # # triplets = np.repeat(np.array(triplets), 5, axis=0)
            # show_triplet_grid(triplets, self, bp)

    ########################
    #     TESTING STATS    #
    ########################

    def qg_pairwise_dist_statistics(self, pairwise_dist, body_part_pairwise_dist, qf_parts_visibility, gf_parts_visibility):
        valid_pairwise_dist_mask = (pairwise_dist != float(-1))
        invalid_pairs_count = (~valid_pairwise_dist_mask).sum()
        self.invalid_pairs_count_at_test_time.update(invalid_pairs_count, valid_pairwise_dist_mask.nelement())
        uncomparable_queries_count = (~valid_pairwise_dist_mask.max(dim=1)[0]).sum()
        self.uncomparable_queries_at_test_time.update(uncomparable_queries_count, valid_pairwise_dist_mask.shape[0])

        part_pairwise_dist_numpy = body_part_pairwise_dist.numpy()
        self.qg_body_part_distances_boxplot(part_pairwise_dist_numpy)
        self.qg_body_part_pairs_availability_barplot(part_pairwise_dist_numpy)
        if qf_parts_visibility is not None and gf_parts_visibility is not None:
            qf_parts_visibility = qf_parts_visibility.numpy()
            gf_parts_visibility = gf_parts_visibility.numpy()
            self.qg_body_part_availability_barplot(qf_parts_visibility, gf_parts_visibility)
            self.qg_distribution_of_body_part_availability_histogram(qf_parts_visibility, gf_parts_visibility)

    def qg_body_part_distances_boxplot(self, body_part_pairwise_dist):
        histogram = body_part_pairwise_dist.reshape(body_part_pairwise_dist.shape[:-2] + (-1,)).transpose()
        idx_to_keep = np.random.choice(histogram.shape[0], min(2000, histogram.shape[0]), replace=False)
        idx_to_keep = np.concatenate([idx_to_keep, np.argmax(histogram, axis=0), np.argmin(histogram, axis=0)])
        sampled_histogram = histogram[idx_to_keep]
        valid_distances_histogram = [sampled_histogram[sampled_histogram[:, i] != -1, i] for i in range(0, sampled_histogram.shape[1])]
        fig, ax = plt.subplots(figsize=(24, 4))
        ax.boxplot(valid_distances_histogram, notch=True, widths=0.35, labels=range(0, body_part_pairwise_dist.shape[0]))
        ax.set_ylabel('Distance')
        ax.set_xlabel('Body part index')
        ax.set_title('Distance distribution of query-gallery body part pairs')
        fig.tight_layout()
        self.logger.add_figure("Query-gallery body part distances boxplot", fig, self.engine_state.global_step)

    def qg_body_part_pairs_availability_barplot(self, body_part_pairwise_dist):
        body_part_pairs_availability = (body_part_pairwise_dist != -1).mean(axis=(1, 2))
        x_labels = range(0, len(body_part_pairs_availability))
        x = np.arange(len(x_labels))  # the label locations
        width = 0.7  # the width of the bars
        fig, ax = plt.subplots(figsize=(24, 4))
        rects = ax.bar(x, body_part_pairs_availability, width)
        # Add some text for x_labels, title and custom x-axis tick x_labels, etc.
        ax.set_ylabel('Availability')
        ax.set_xlabel('Body part index')
        ax.set_title('Query-gallery body parts pairs availability')
        ax.set_yticks(np.arange(0, 1.2, 0.1))
        ax.yaxis.get_major_ticks()[-1].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}%'.format(int(height*100)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects)

        fig.tight_layout()
        self.logger.add_figure("Query-gallery body part pairs availability barplot", fig, self.engine_state.global_step)

    def qg_body_part_availability_barplot(self, qf_parts_visibility, gf_parts_visibility):
        qf_mask_availability = qf_parts_visibility.mean(axis=0)
        gf_mask_availability = gf_parts_visibility.mean(axis=0)
        x_labels = range(0, len(qf_mask_availability))
        x = np.arange(len(x_labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(figsize=(24, 4))
        rects1 = ax.bar(x - width / 2, qf_mask_availability, width, label='Query')
        rects2 = ax.bar(x + width / 2, gf_mask_availability, width, label='Gallery')
        # Add some text for x_labels, title and custom x-axis tick x_labels, etc.
        ax.set_ylabel('Availability')
        ax.set_xlabel('Body part index')
        ax.set_title('Body parts availability for {} query and {} gallery samples'.format(qf_parts_visibility.shape[0], gf_parts_visibility.shape[0]))
        ax.set_yticks(np.arange(0, 1.2, 0.1))
        ax.yaxis.get_major_ticks()[-1].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.legend()

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}%'.format(int(height*100)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        # autolabel(rects1)
        # autolabel(rects2)
        fig.tight_layout()
        self.logger.add_figure("Query-gallery body part availability barplot", fig, self.engine_state.global_step)

    def qg_distribution_of_body_part_availability_histogram(self, qf_parts_visibility, gf_parts_visibility):
        qf_mask_availability = qf_parts_visibility.sum(axis=1)
        gf_mask_availability = gf_parts_visibility.sum(axis=1)

        x = np.arange(gf_parts_visibility.shape[1]+2)
        qf_mask_availability_distribution = np.histogram(qf_mask_availability, bins=x)[0]/len(qf_parts_visibility)
        gf_mask_availability_distribution = np.histogram(gf_mask_availability, bins=x)[0]/len(gf_parts_visibility)

        x_labels = np.arange(gf_parts_visibility.shape[1]+1)
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots(figsize=(24, 4))
        ax.bar(x_labels - width / 2, qf_mask_availability_distribution, width, label='Query')
        ax.bar(x_labels + width / 2, gf_mask_availability_distribution, width, label='Gallery')
        # Add some text for x_labels, title and custom x-axis tick x_labels, etc.
        ax.set_ylabel('Samples count')
        ax.set_xlabel('Amount of body parts available')
        ax.set_title('Body parts availability distribution for {} query and {} gallery samples'.format(qf_parts_visibility.shape[0], gf_parts_visibility.shape[0]))
        # ax.set_yticks(np.arange(0, 1.2, 0.1))
        ax.yaxis.get_major_ticks()[-1].set_visible(False)
        ax.set_xticks(x_labels)
        ax.set_xticklabels(x_labels)
        ax.legend()

        fig.tight_layout()
        self.logger.add_figure("Query-gallery distribution of body part availability histogram", fig, self.engine_state.global_step)

    def visualize_embeddings(self, qf, gf, q_pids, g_pids, test_loader, dataset_name, qf_parts_visibility, gf_parts_visibility, mAP, rank1):
        if self.cfg.test.vis_embedding_projection and not self.engine_state.is_training:
            print("Visualizing embeddings projection")
            visualize_embeddings(qf, gf, q_pids, g_pids, test_loader, dataset_name, qf_parts_visibility, gf_parts_visibility, mAP, rank1)

    def visualize_rank(self, test_loader, dataset_name, distmat, save_dir, visrank_topk, visrank_q_idx_list, visrank_count,
                        body_parts_distmat, qf_parts_visibility, gf_parts_visibility, q_parts_masks, g_parts_masks, mAP, rank1):
        if self.cfg.test.visrank:
            save_dir = os.path.join(save_dir, 'vis_bp_rank_' + dataset_name)
            visualize_ranking_grid(distmat, body_parts_distmat, test_loader, dataset_name,
                           qf_parts_visibility, gf_parts_visibility, q_parts_masks, g_parts_masks, mAP, rank1, save_dir, visrank_topk, visrank_q_idx_list, visrank_count, config=self.cfg)

            if self.cfg.test.visrank_per_body_part:
                for bp in range(0, body_parts_distmat.shape[0]):
                    qf_part_visibility = None
                    if qf_parts_visibility is not None:
                        qf_part_visibility = qf_parts_visibility[:, bp:bp+1]
                    gf_part_visibility = None
                    if gf_parts_visibility is not None:
                        gf_part_visibility = gf_parts_visibility[:, bp:bp+1]
                    visualize_ranking_grid(body_parts_distmat[bp], body_parts_distmat[bp:bp+1], test_loader, dataset_name,
                                   qf_part_visibility, gf_part_visibility, q_parts_masks, g_parts_masks, mAP, rank1, save_dir, visrank_topk, visrank_q_idx_list, visrank_count, config=self.cfg, bp_idx=bp)

    ########################
    #    RUNNING EVENTS    #
    ########################

    def training_started(self):
        self.report_performance([0, 0, 0, 0], 0, 0, 0)

    def epoch_started(self):
        self.logger.add_scalar('Other/epoch', self.engine_state.epoch, self.engine_state.global_step)
        self.logger.add_scalar('Other/batch', self.engine_state.batch, self.engine_state.global_step)
        self.logger.add_scalar('Other/iteration', self.engine_state.global_step, self.engine_state.global_step)

    def epoch_completed(self):
        eta_seconds = (self.max_epoch - (self.engine_state.epoch + 1)) * self.epoch_timer.avg / 1000
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(
            'epoch: [{0}/{1} e][{2} b]\t'
            'eta {eta}\t'
            'lr {lr:.8f}\t'
            'loss {loss:.3f}\t'
            '{losses}'.format(
                self.engine_state.epoch + 1,
                self.max_epoch,
                self.engine_state.batch,
                eta=eta_str,
                lr=self.engine_state.lr,
                loss=self.loss.epoch_ratio(self.engine_state.epoch),
                losses=self.losses.summary(self.engine_state.epoch)
            )
        )

        for name, dict in self.losses.meters.items():
            for key, meter in dict.items():
                self.logger.add_scalar('Loss/' + name + "_" + key + '_avg', meter.mean[self.engine_state.epoch], self.engine_state.epoch)

        # self.logger.add_scalar("Training/Trivial triplets in batch", self.zero_losses_count.epoch_ratio(self.engine_state.epoch), self.engine_state.epoch)

        if not self.used_body_parts_in_max.is_empty:
            for bp_id, bp_ratio in enumerate(self.used_body_parts_in_max.epoch_ratio(self.engine_state.epoch)):
                self.logger.add_scalar("Used body parts in training/bp {}".format(bp_id), bp_ratio, self.engine_state.epoch)

    def training_completed(self):
        print("Training completed")
        # TODO fix metrics affected by loss refactor
        print("Average image pairs that couldn't be compared within one batch: {}%".format(perc(self.invalid_pairwise_distances_count.total_ratio())))
        print("Average body part pairs that couldn't be compared within one batch: {}%".format(perc(self.uncomparable_body_parts_pairs_count.total_ratio())))
        # print("Average valid triplets during one epoch: {}%".format(perc(self.valid_triplets_mask_count.total_ratio())))
        self.display_used_body_parts()

    def test_completed(self):
        print("Test completed")
        if not self.invalid_pairs_count_at_test_time.is_empty:
            print("Amount of pairs query-gallery that couldn't be compared: {}%".format(perc(self.invalid_pairs_count_at_test_time.ratio(), 3)))
        if not self.uncomparable_queries_at_test_time.is_empty:
            print("Amount of queries that couldn't be compared to any gallery sample: {}%".format(perc(self.uncomparable_queries_at_test_time.ratio(), 3)))

    def run_completed(self):
        timer_meters = [
                        self.total_run_timer,
                        self.epoch_timer,
                        self.batch_timer,
                        self.data_loading_timer,
                        self.feature_extraction_timer,
                        self.loss_timer,
                        self.optimizer_timer,
                        self.performance_evaluation_timer,
                        self.test_timer,
                        self.test_batch_timer,
                        ]
        table = []
        for time_meter in timer_meters:
            table.append([time_meter.name, time_meter.average_time(), time_meter.total_time(), time_meter.count])
        headers = ["Time metric name", "Average", "Total", "Count"]

        print(tabulate(table, headers, tablefmt="fancy_grid"))

    ########################
    #        UTILS         #
    ########################

    def display_used_body_parts(self):
        if self.used_body_parts_in_max.is_empty:
            return

        # plot histogram
        body_parts_used_for_training = self.used_body_parts_in_max.total_ratio()
        x_labels = range(0, len(body_parts_used_for_training))
        x = np.arange(len(x_labels))  # the label locations
        width = 0.7  # the width of the bars
        fig, ax = plt.subplots(figsize=(24, 4))
        rects = ax.bar(x, body_parts_used_for_training, width)
        # Add some text for x_labels, title and custom x-axis tick x_labels, etc.
        ax.set_ylabel('Selection percentage')
        ax.set_xlabel('Body part index')
        ax.set_title('Body parts used for training')
        ax.set_yticks(np.arange(0, 1.2, 0.1))
        ax.yaxis.get_major_ticks()[-1].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        def autolabel(rects):
            """Attach a text label above each bar in *rects*, displaying its height."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}%'.format(np.around(height*100, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects)

        fig.tight_layout()
        self.logger.add_figure("Body parts used for training", fig, self.engine_state.global_step)
