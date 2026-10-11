import torch
from torch.nn import functional as F

from detectron2.utils.comm import get_world_size

from .criterion import SetCriterion, morphological_boundary
from .tree_geometry import resize_padding_mask
from ..utils.misc import is_dist_avail_and_initialized


class TreeAwareSetCriterion(SetCriterion):
    def __init__(self, *args, center_sigma=2.0, separation_margin=0.2,
                 separation_radius=0.15, positive_weight=1.0, negative_weight=1.0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if center_sigma <= 0:
            raise ValueError('CENTER_SIGMA must be positive')
        if not -1 <= separation_margin <= 1:
            raise ValueError('SEPARATION_MARGIN must be in [-1, 1]')
        self.center_sigma = center_sigma
        self.separation_margin = separation_margin
        self.separation_radius = separation_radius
        self.positive_weight = positive_weight
        self.negative_weight = negative_weight

    def loss_masks(self, outputs, targets, indices, num_masks):
        # PointRend sampling does not need to run for an empty assignment.
        if sum(src.numel() for src, _ in indices) == 0:
            zero = outputs['pred_masks'].sum() * 0.0
            return {'loss_mask': zero, 'loss_dice': zero}
        return super().loss_masks(outputs, targets, indices, num_masks)

    def loss_boundary(self, outputs, targets, indices, num_masks):
        logits = outputs['pred_masks']
        total = logits.sum() * 0.0
        size = logits.shape[-2:]
        for batch, (src, tgt) in enumerate(indices):
            if src.numel() == 0:
                continue
            src = src.to(logits.device)
            tgt = tgt.to(targets[batch]['masks'].device)
            valid = ~resize_padding_mask(targets[batch]['padding_mask'][None], size)[0]
            predicted = logits[batch, src, None].float().sigmoid() * valid
            expected = F.interpolate(
                targets[batch]['masks'][tgt, None].float(), size, mode='nearest'
            ) * valid
            predicted = (morphological_boundary(predicted) * valid).flatten(1)
            expected = (morphological_boundary(expected) * valid).flatten(1)
            loss = 1 - (2 * (predicted * expected).sum(1) + 1) / (
                predicted.sum(1) + expected.sum(1) + 1
            )
            total = total + (loss * (expected.sum(1) > 0)).sum()
        return {'loss_boundary': total / num_masks}

    def loss_geometry(self, outputs, targets):
        maps = outputs['tree_geometry']
        geometry = [target['tree_geometry'] for target in targets]
        valid = torch.stack([target['valid'] for target in geometry]).float()
        result = {}

        if self.weight_dict.get('loss_tree_center', 0) > 0:
            target = torch.stack([item['center'] for item in geometry])
            p = maps['center_logits'][:, 0].float().sigmoid().clamp(1e-4, 1 - 1e-4)
            positive = (target == 1).float() * valid
            negative = (target < 1).float() * valid
            pos_loss = -p.log() * (1 - p).pow(2) * positive
            neg_loss = -(1 - p).log() * p.pow(2) * (1 - target).pow(4) * negative
            pos_count = positive.sum((1, 2))
            denominator = torch.where(pos_count > 0, pos_count, valid.sum((1, 2))).clamp_min(1)
            result['loss_tree_center'] = ((pos_loss + neg_loss).sum((1, 2)) / denominator).mean()

        if self.weight_dict.get('loss_tree_boundary', 0) > 0:
            target = torch.stack([item['boundary'] for item in geometry])
            logits = maps['boundary_logits'][:, 0].float()
            pos = (target * valid).sum()
            pos_weight = ((valid.sum() - pos) / pos.clamp_min(1)).clamp(1, 20)
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            bce = (bce * torch.where(target > 0, pos_weight, 1.0) * valid).sum() / valid.sum().clamp_min(1)
            predicted = logits.sigmoid() * valid
            numerator = 2 * (predicted * target).sum((1, 2)) + 1
            denominator = predicted.sum((1, 2)) + (target * valid).sum((1, 2)) + 1
            result['loss_tree_boundary'] = bce + (1 - numerator / denominator).mean()

        if self.weight_dict.get('loss_tree_distance', 0) > 0:
            target = torch.stack([item['distance'] for item in geometry])
            foreground = torch.stack([item['foreground'] for item in geometry]) * valid
            background = (1 - foreground) * valid
            predicted = maps['distance_logits'][:, 0].float().sigmoid()
            error = F.smooth_l1_loss(predicted, target, reduction='none')
            fg_loss = (error * foreground).sum() / foreground.sum().clamp_min(1)
            bg_loss = (error * background).sum() / background.sum().clamp_min(1)
            result['loss_tree_distance'] = fg_loss + 0.1 * bg_loss
        return result

    def loss_separation(self, outputs, indices, aux_outputs, aux_indices, targets):
        features = outputs['query_features'].float()
        zero = features.sum() * 0.0
        positive_total = zero
        negative_total = zero
        positive_count = 0
        negative_count = 0
        # Each state is matched independently; query slot IDs need not coincide.
        states = list(zip(aux_outputs, aux_indices)) + [(outputs, indices)]
        for batch, (src, tgt) in enumerate(indices):
            src = src.to(features.device)
            tgt = tgt.to(features.device)
            matched = F.normalize(features[batch, src], dim=-1)
            if src.numel() >= 2:
                similarity = matched @ matched.transpose(0, 1)
                centers = targets[batch]['tree_centroids'][tgt]
                neighbors = torch.cdist(centers.float(), centers.float()) <= self.separation_radius
                pairs = torch.triu(neighbors, diagonal=1)
                negative_total = negative_total + F.relu(
                    similarity[pairs] - self.separation_margin
                ).sum()
                negative_count += pairs.sum().item()
            # Positive pairs are the same GT instance in adjacent decoder states.
            for (left, left_indices), (right, right_indices) in zip(states[:-1], states[1:]):
                left_src, left_tgt = left_indices[batch]
                right_src, right_tgt = right_indices[batch]
                left_src, left_tgt = left_src.to(features.device), left_tgt.to(features.device)
                right_src, right_tgt = right_src.to(features.device), right_tgt.to(features.device)
                pairs = (left_tgt[:, None] == right_tgt[None, :]).nonzero(as_tuple=False)
                if pairs.numel() == 0:
                    continue
                a = F.normalize(left['query_features'][batch, left_src[pairs[:, 0]]].float(), dim=-1)
                b = F.normalize(right['query_features'][batch, right_src[pairs[:, 1]]].float(), dim=-1)
                positive_total = positive_total + (1 - (a * b).sum(-1)).sum()
                positive_count += pairs.shape[0]
        return {'loss_query_separation': (
            self.positive_weight * positive_total / max(positive_count, 1)
            + self.negative_weight * negative_total / max(negative_count, 1)
        )}

    def forward(self, outputs, targets):
        final = {key: value for key, value in outputs.items() if key != 'aux_outputs'}
        indices = self.matcher(final, targets)
        num_masks = torch.tensor(
            [sum(len(target['labels']) for target in targets)],
            dtype=torch.float32, device=outputs['pred_logits'].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = (num_masks / get_world_size()).clamp_min(1).item()
        result = {}
        for loss in self.losses:
            result.update(self.get_loss(loss, final, targets, indices, num_masks))
        if self.weight_dict.get('loss_boundary', 0) > 0:
            result.update(self.loss_boundary(final, targets, indices, num_masks))
        result.update(self.loss_geometry(final, targets))
        auxiliary = outputs.get('aux_outputs', [])
        auxiliary_indices = []
        for layer, prediction in enumerate(auxiliary):
            assignment = self.matcher(prediction, targets)
            auxiliary_indices.append(assignment)
            for loss in self.losses:
                values = self.get_loss(loss, prediction, targets, assignment, num_masks)
                result.update({f'{key}_{layer}': value for key, value in values.items()})
        if self.weight_dict.get('loss_query_separation', 0) > 0:
            result.update(self.loss_separation(final, indices, auxiliary, auxiliary_indices, targets))
        return result
