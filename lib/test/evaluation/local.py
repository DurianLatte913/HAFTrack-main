from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.lmdb_path=""
    
    settings.gtot_path = '/opt/data/VTUAV/GTOT'
    settings.lasher_path = "./data/LasHeR/"
    settings.lashertestingSet_path = './data/LasHeR/test/'
    settings.lasher_unaligned = ""
    
    settings.network_path = ''    # 预训练权重的路径
    settings.results_path = './tracking_result'
    
    settings.rgbt210_path = './data/RGBT210/'
    settings.rgbt234_path = "./data/RGBT234/"
    settings.vtuav_path = "./data/VTUAV/"
    settings.gtot_dir = '/opt/data/VTUAV/GTOT'
    
    settings.segmentation_path = ''

    settings.save_dir="./output/"
    settings.prj_dir = "./"


    return settings

