from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.lmdb_path="/home/zhaojiacong/datasets/lmdb_dataset/"
    
    settings.gtot_path = '/opt/data/VTUAV/GTOT'
    settings.lasher_path = "/opt/data/VTUAV/lasher/"
    settings.lashertestingSet_path = '/opt/data/VTUAV/lasher/test/'
    settings.lasher_unaligned = "/home/zhaojiacong/datasets/LasHeR_Ualigned/"
    
    # settings.network_path = '/home/zhaojiacong/all_pretrained/tomp/author/'    # Where tracking networks are stored.  # 预训练权重的路径
    settings.network_path = '/home/zhaojiacong/all_pretrained/trained_by_me/'    # 预训练权重的路径
    settings.results_path = './tracking_result'
    # settings.result_plot_path = '/home/zhaojiacong/all_result/lasher_test/'
    # settings.result_plot_path = '/home/zhaojiacong/all_result/rgbt234/'
    
    settings.rgbt210_path = '/opt/data/VTUAV/RGBT210/'
    settings.rgbt234_path = "/opt/data/VTUAV/RGBT234/"
    settings.vtuav_path = "/opt/data/VTUAV/vtuav_all/"
    settings.gtot_dir = '/opt/data/VTUAV/GTOT'
    
    settings.segmentation_path = '/data/liulei/pytracking/pytracking/segmentation_results/'

    settings.save_dir="./output/"
    settings.prj_dir = "./"


    return settings

