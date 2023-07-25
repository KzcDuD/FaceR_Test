# FaceR_Test

Using transform learning to implement face recognition

+ choose `torchvision.models` to pretrain.

    1. `ResNet`: ResNet 是一個非常流行的深度殘差網絡，具有 18 層、34 層、50 層、101 層等不同版本。ResNet 可以處理較深的網絡，並且在許多圖像分類任務中表現良好。

    1. `VGG`: VGG 是一個經典的卷積神經網絡，具有較小的濾波器大小 (3x3) 和更深的網絡結構。VGG 簡單易懂，而且在小數據集上表現良好。

    1. `DenseNet`: DenseNet 是一個密集連接的網絡結構，它允許特徵的高度重用，並且在參數較少的情況下具有較高的準確率。

    1. `MobileNet`: MobileNet 是一個輕量級的網絡結構，它專注於在資源有限的設備上實現高效的運行速度。

    1. `Inception`: Inception 系列模型具有多種卷積核大小和不同層次的濾波器，這有助於捕捉不同尺度下的特徵。

    ---

    1. `資料集大小`：如果您的資料集很大，您可以考慮使用複雜的模型，例如 ResNet-50 或更深的版本。如果資料集較小，則選擇一個較小的模型，例如 VGG 或 MobileNet，可以更好地避免過擬合。

    2. `任務需求`：不同的模型可能對不同的任務有較好的適應性。例如，某些模型在分類任務上效果較好，而某些模型在物體檢測或分割等任務上效果較好。

    3. `計算資源`：複雜的模型需要更多的計算資源和記憶體，而較小的模型則需要更少。確保您有足夠的計算資源來支持選擇的模型。

    4. `模型特徵`：不同模型可能對於不同尺寸和類型的特徵有著不同的提取能力。請確保選擇的模型能夠適合您的任務特點。

1. `freeze`: accuracy is low
1. `Finetuning`

    |   |Resnet-18|Resnet-34|Resnet-50|
    |---|---------|---------|---------|
    |Acc|65.5172%|79.310345%|65.517241%|

## Reference

[face_recognition Github](https://github.com/ageitgey/face_recognition)

[how-to-detect-facial-recognition](https://khemsok97.medium.com/how-to-detect-facial-recognition-with-transfer-learning-and-pytorch-16e3e95c9cd7)
