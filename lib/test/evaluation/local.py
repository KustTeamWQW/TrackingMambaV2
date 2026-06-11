from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/zly/projects/datasets/got10k_lmdb'
    settings.got10k_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/itb'
    settings.lasot_extension_subset_path_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/lasot_lmdb'
    settings.lasot_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/lasot'
    settings.network_path = '/home/zly/projects/pythonprojects/TrackingmambaV2/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/nfs'
    settings.otb_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/otb'
    settings.prj_dir = '/home/zly/projects/pythonprojects/TrackingmambaV2'
    settings.result_plot_path = '/home/zly/projects/pythonprojects/TrackingmambaV2/output/test/result_plots'
    settings.results_path = '/home/zly/projects/pythonprojects/TrackingmambaV2/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/zly/projects/pythonprojects/TrackingmambaV2/output'
    settings.segmentation_path = '/home/zly/projects/pythonprojects/TrackingmambaV2/output/test/segmentation_results'
    settings.tc128_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/trackingnet'

    settings.vot18_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/vot2018'
    settings.vot22_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/vot2022'
    settings.vot_path = '/home/z/Documents/GraduationThesis/github/TrackingMambaV2/data/VOT2019'
    settings.uav_path = '/home/zly/projects/datasets/UAV123'
    settings.dtb70_path = '/home/zly/projects/datasets/DTB70'
    settings.otmj_path = '/home/zly/projects/datasets/OTMJ'
    settings.otmj_tir_path = '/home/zly/projects/datasets/OTMJ_TIR'
    settings.otmj_cross_path = '/home/zly/projects/datasets/OTMJ_cross'
    settings.otmj_cross_random_path = '/home/zly/projects/datasets/OTMJ_cross_random'
    settings.youtubevos_dir = ''
    settings.ptbtir_path = '/home/zly/projects/datasets/ptbtir'

    return settings
