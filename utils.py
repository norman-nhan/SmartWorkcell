import pyrealsense2 as rs
from groundingdino.util.inference import Model
from nanosam.utils.predictor import Predictor

def load_models(gdino_checkpoint="/home/non/nhan_ws/src/SmartWorkcell/io/weights/groundingdino_swinb_cogcoor.pth",
                gdino_config="/home/non/nhan_ws/src/SmartWorkcell/io/config/gdino/GroundingDINO_SwinB_cfg.py",
                sam_image_encoder="/home/non/nhan_ws/src/SmartWorkcell/io/weights/resnet18_image_encoder.engine",
                sam_mask_decoder="/home/non/nhan_ws/src/SmartWorkcell/io/weights/mobile_sam_mask_decoder.engine"):
    # load gdino
    gdino_model = Model(
        model_checkpoint_path=gdino_checkpoint,
        model_config_path=gdino_config
    )
    # load nanosam predictor
    sam_predictor = Predictor(
        image_encoder_engine=sam_image_encoder,
        mask_decoder_engine=sam_mask_decoder
    )
    return gdino_model, sam_predictor

def realsense_setup():
    pipeline = rs.pipeline()
    config = rs.config()
    colorizer = rs.colorizer()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    return pipeline, config, colorizer
