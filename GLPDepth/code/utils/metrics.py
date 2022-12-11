import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape

    mask_20 = target < 20
    mask_80 = target < 80
    mask_20_80 = torch.logical_and(target > 20, target < 80)
    mask_all = target > 0.001
    mask_80_256 = target > 80
    masks = [mask_20, mask_80, mask_20_80, mask_80_256, mask_all]
    mask_names = [
        'mask_20',
        'mask_80',
        'mask_20_80',
        'mask_80_256',
        'mask_all']

    all_results = {}

    for i, mask in enumerate(masks):

        if torch.sum(mask.long()) < 1:
            continue

        thresh = torch.max(
            (target[mask] / pred[mask]),
            (pred[mask] / target[mask]))

        d1 = torch.sum(thresh < 1.25).float() / len(thresh)  # d1
        d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)  # d2
        d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)  # d3

        diff = pred[mask] - target[mask]

        diff_log = torch.log(pred[mask]) - torch.log(target[mask])

        abs_rel = torch.mean(torch.abs(diff) / target[mask])  # abs_rel

        sq_rel = torch.mean(torch.pow(diff, 2) / target[mask])  # sq_rel

        rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))  # rmse

        rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))  # rmse_log

        log10 = torch.mean(
            torch.abs(
                torch.log10(
                    pred[mask]) -
                torch.log10(
                    target[mask])))  # log_10

        silog = torch.sqrt(
            torch.pow(
                diff_log,
                2).mean() -
            0.5 *
            torch.pow(
                diff_log.mean(),
                2))  # silog

        results_dict = {
            'd1': d1.item(),
            'd2': d2.item(),
            'd3': d3.item(),
            'abs_rel': abs_rel.item(),
            'sq_rel': sq_rel.item(),
            'rmse': rmse.item(),
            'rmse_log': rmse_log.item(),
            'log10': log10.item(),
            'silog': silog.item(),
            'iter': 1}

        all_results[mask_names[i]] = results_dict

    return all_results


'''
            'd1_mask_20': d1_mask_20.item(), 'd2_mask_20': d2_mask_20.item(), 'd3_mask_20': d3_mask_20.item(), 'abs_rel_mask_20': abs_rel_mask_20.item(),
            'sq_rel_mask_20': sq_rel.item(), 'rmse_mask_20': rmse.item(), 'rmse_log_mask_20': rmse_log_mask_20.item(),
            'log10_mask_20':log10_mask_20.item(), 'silog_mask_20':silog_mask_20.item(), 'd1_mask_80': d1_mask_80.item(), 'd2_mask_80': d2_mask_80.item(), 'd3_mask_80': d3_mask_80.item(), 'abs_rel_mask_80': abs_rel_mask_80.item(),
            'sq_rel_mask_80': sq_rel_mask_80.item(), 'rmse_mask_80': rmse_mask_80.item(), 'rmse_log_mask_80': rmse_log_mask_80.item(),
            'log10_mask_80':log10_mask_80.item(), 'silog_mask_80':silog_mask_80.item(), 'd1_mask_20_80': d1_mask_20_80.item(), 'd2_mask_20_80': d2_mask_20_80.item(), 'd3_mask_20_80': d3_mask_20_80.item(), 'abs_rel_mask_20_80': abs_rel_mask_20_80.item(),
            'sq_rel_mask_20_80': sq_rel_mask_20_80.item(), 'rmse_mask_20_80': rmse_mask_20_80.item(), 'rmse_log_mask_20_80': rmse_log_mask_20_80.item(),
            'log10_mask_20_80':log10_mask_20_80.item(), 'silog_mask_20_80':silog_mask_20_80.item(), 'd1_mask_256': d1_mask_256.item(), 'd2_mask_256': d2_mask_256.item(), 'd3_mask_256': d3_mask_256.item(), 'abs_rel_mask_256': abs_rel_mask_256.item(),
            'sq_rel_mask_256': sq_rel_mask_256.item(), 'rmse_mask_256': rmse_mask_256.item(), 'rmse_log_mask_256': rmse_log_mask_256.item(),
            'log10_mask_256':log10_mask_256.item(), 'silog_mask_256':silog_mask_256.item()}

'''


def cropping_img(args, pred, gt_depth):
    min_depth_eval = args.min_depth_eval

    max_depth_eval = args.max_depth_eval

    pred[torch.isinf(pred)] = max_depth_eval
    pred[torch.isnan(pred)] = min_depth_eval

    valid_mask = torch.logical_and(
        gt_depth > min_depth_eval, gt_depth < max_depth_eval)

    if args.dataset == 'kitti':
        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_depth = gt_depth[top_margin:top_margin +
                                352, left_margin:left_margin + 1216]

        if args.kitti_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = torch.zeros(valid_mask.shape).to(
                device=valid_mask.device)

            if args.kitti_crop == 'garg_crop':
                eval_mask[int(0.40810811 *
                              gt_height):int(0.99189189 *
                                             gt_height), int(0.03594771 *
                                                             gt_width):int(0.96405229 *
                                                                           gt_width)] = 1

            elif args.kitti_crop == 'eigen_crop':
                eval_mask[int(0.3324324 *
                              gt_height):int(0.91351351 *
                                             gt_height), int(0.0359477 *
                                                             gt_width):int(0.96405229 *
                                                                           gt_width)] = 1
            else:
                eval_mask = valid_mask

    elif args.dataset == 'nyudepthv2':
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[45:471, 41:601] = 1

    elif args.dataset == 'cityscapes':
        gt_height, gt_width = gt_depth.shape
        eval_mask = torch.zeros(valid_mask.shape).to(device=valid_mask.device)
        eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                  int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

    else:
        eval_mask = valid_mask

    valid_mask = torch.logical_and(valid_mask, eval_mask)

    return pred[valid_mask], gt_depth[valid_mask]
