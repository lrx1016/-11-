import torch
import segmentation_models_pytorch as smp

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = smp.Unet(
    encoder_name="resnet18",
    encoder_weights=None,
    in_channels=3,
    classes=1
).to(device)

model.eval()

x = torch.randn(1, 3, 288, 384).to(device)

with torch.no_grad():

    features = model.encoder(x)

    print("Encoder 特征：")

    for i, feature in enumerate(features):
        print(i, feature.shape)

    # Decoder 接收 Encoder 的多尺度特征
    decoder_output = model.decoder(features)

    print("\nDecoder 输出：")
    print(decoder_output.shape)

    # 分割头生成最终预测
    prediction = model.segmentation_head(
        decoder_output
    )

    print("\n最终预测：")
    print(prediction.shape)
