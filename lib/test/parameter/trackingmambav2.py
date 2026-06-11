from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.trackingmambav2.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/trackingmambav2/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    epoch_override = os.environ.get("TRACKINGMAMBAV2_TEST_EPOCH")
    if epoch_override:
        cfg.TEST.EPOCH = int(epoch_override)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    checkpoint = os.path.join(save_dir, "checkpoints/train/trackingmambav2/%s/TrackingmambaV2_ep%04d.pth.tar" %
                              (yaml_name, cfg.TEST.EPOCH))
    if not os.path.isfile(checkpoint):
        legacy_checkpoint = os.path.join(save_dir, "checkpoints/train/aqatrack/%s/AQATrack_ep%04d.pth.tar" %
                                         (yaml_name.replace("TrackingmambaV2", "AQATrack"), cfg.TEST.EPOCH))
        if os.path.isfile(legacy_checkpoint):
            checkpoint = legacy_checkpoint
    params.checkpoint = checkpoint

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
