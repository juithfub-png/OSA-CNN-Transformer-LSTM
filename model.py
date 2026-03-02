##导入模块部分
import pickle
import keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Add, Dense, Dropout, LSTM, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.models import Model  # 修正导入语句
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from scipy.interpolate import splev, splrep
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

base_dir = "dataset"#数据集目录
ir = 3  # interpolate interval#插值间隔，控制插值密度
before = 2 #事件发生前分钟数
after = 2 #事件发生后分钟数

# normalize归一化处理
scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) #归一化函数(lambda)：将数值缩放到[0,1]范围内，公式为：对数组中的每个元素，减去数组的最小值，然后除以数组的极差（最大值减最小值）


def load_data_with_analysis(): #负责数据加载及预处理函数
    """加载数据并进行详细分析"""
    try:
        tm = np.arange(0, (before + 1 + after) * 60, step=1 / float(ir))
        #创建时间轴：从0到(before+1+after)分钟，步长为1/ir
        with open(os.path.join(base_dir, "apnea-ecg.pkl"), 'rb') as f:
            apnea_ecg = pickle.load(f)
	#加载pickle格式的数据：在Python中，pickle是一种用于序列化和反序列化的模块。所谓pickle格式的数据，就是通过Python的pickle模块将对象序列化后得到的字节流数据。这种格式的数据可以将Python对象（如列表、字典、类实例等）保存到文件中，或者通过网络传输，然后在需要的时候恢复成原来的对象。
  #序列化：将python对象转换成字节流的过程
        #初始化训练数据集
x_train = []
#提取原始训练数据及标签
        o_train, y_train = apnea_ecg["o_train"], apnea_ecg["y_train"]
        groups_train = apnea_ecg["groups_train"]#训练集患者分组
        #分析训练数据
        print("=" * 60)
        print("train_data_analysis")
        print("=" * 60)
        # 检查原始标签分布
        y_train_np = np.array(y_train)
        print(f"训练集样本总数: {len(y_train_np)}")
        print(f"正常样本数: {np.sum(y_train_np == 0)}")#统计标签为0的样本数量
        print(f"呼吸暂停样本数: {np.sum(y_train_np == 1)}")#统计标签为1的样本数量
        print(f"呼吸暂停比例: {np.sum(y_train_np == 1) / len(y_train_np):.4f}")
	#处理每个训练样本
        for i in range(len(o_train)):
            #o_train包含两个信号：RRI（RR间期）和幅度
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_train[i]
            # Curve interpolation——使用样条插值将信号插值到统一的时间轴
            rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
            ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
            x_train.append([rri_interp_signal, ampl_interp_signal])
	       #splrep函数：创建样条表示
	       #splev函数：在指定点评估样条

#转换数据形状：原始形状：(样本数，两个信号，时间点)，transpose后：(样本数，时间点，2个信号)
        x_train = np.array(x_train, dtype="float32").transpose((0, 2, 1))
        y_train = np.array(y_train, dtype="float32")

        print(f"训练数据形状: {x_train.shape}")
        print(f"训练标签形状: {y_train.shape}")
	#处理测试数据集（具体内容同处理训练集数据）
        x_test = []
        o_test, y_test = apnea_ecg["o_test"], apnea_ecg["y_test"]
        groups_test = apnea_ecg["groups_test"]
	#分析测试数据
        print("\n" + "=" * 60)
        print("测试数据分析")
        print("=" * 60)
        # 检查测试标签分布
        y_test_np = np.array(y_test)
        print(f"测试集样本总数: {len(y_test_np)}")
        print(f"正常样本数: {np.sum(y_test_np == 0)}")
        print(f"呼吸暂停样本数: {np.sum(y_test_np == 1)}")
        print(f"呼吸暂停比例: {np.sum(y_test_np == 1) / len(y_test_np):.4f}")
	#处理每个测试样本
        for i in range(len(o_test)):
            (rri_tm, rri_signal), (ampl_tm, ampl_signal) = o_test[i]
            rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
            ampl_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
            x_test.append([rri_interp_signal, ampl_interp_signal])
	#转换数据形状
        x_test = np.array(x_test, dtype="float32").transpose((0, 2, 1))
        y_test = np.array(y_test, dtype="float32")

        print(f"测试数据形状: {x_test.shape}")
        print(f"测试标签形状: {y_test.shape}")

        # 添加数据标准化
          重塑数据以便标准化：将3D数据展平为2D
        n_samples, n_timesteps, n_features = x_train.shape
        x_train_reshaped = x_train.reshape(-1, n_features)#形状：(样本数×时间点，特征数)
        x_test_reshaped = x_test.reshape(-1, n_features)
        # 标准化
        scaler_std = StandardScaler()#标准化函数：标准化数据(零均值，单位方差)
z = (x -μ) /σ
        x_train_scaled = scaler_std.fit_transform(x_train_reshaped).reshape(n_samples, n_timesteps, n_features)
        x_test_scaled = scaler_std.transform(x_test_reshaped).reshape(x_test.shape[0], n_timesteps, n_features)

        return x_train_scaled, y_train, groups_train, x_test_scaled, y_test, groups_test
#异常处理：如果加载失败，创建虚拟数据
    except Exception as e:
        print(f"数据加载错误: {e}")
        import traceback
        traceback.print_exc()
        return create_dummy_data()

def create_dummy_data():
    """创建虚拟数据"""
    n_train = 10000
    n_test = 2000
    timesteps = 900

    x_train = np.random.randn(n_train, timesteps, 2).astype("float32")
    y_train = np.random.randint(0, 2, n_train).astype("float32")
    groups_train = np.arange(n_train)

    x_test = np.random.randn(n_test, timesteps, 2).astype("float32")
    y_test = np.random.randint(0, 2, n_test).astype("float32")
    groups_test = np.arange(n_test)

    print(f"使用虚拟数据:")
    print(f"  训练集: {x_train.shape}")
    print(f"  测试集: {x_test.shape}")

    return x_train, y_train, groups_train, x_test, y_test, groups_test


def transformer_encoder_block(inputs, num_heads=2, key_dim=32, dropout_rate=0.5):
    """修正后的Transformer编码器块"""
    # 获取输入形状
    input_shape = tf.keras.backend.int_shape(inputs)
    seq_length = input_shape[1]  # 使用静态形状：序列长度(时间点数量)
    d_model = input_shape[-1]  # 特征维度
    # Layer Normalization 层归一化
    normalized_input = LayerNormalization()(inputs)
    # 创建位置编码（让模型知道序列中元素的位置）
    position = tf.range(seq_length, dtype=tf.float32)
    position = tf.reshape(position, [1, -1, 1])  # 形状: (1, seq_len, 1)
    # 创建角度变化率：频率随维度增加而降低
    angle_rates = 1 / tf.pow(10000.0, 2.0 * tf.range(0, d_model, 2, dtype=tf.float32) / d_model)
    angle_rates = tf.reshape(angle_rates, [1, 1, -1])  # 形状: (1, 1, d_model/2)
    # 计算角度
    angle_rads = position * angle_rates  # 形状: (1, seq_len, d_model/2)
    # 创建正弦和余弦部分
    sin_part = tf.sin(angle_rads)
    cos_part = tf.cos(angle_rads)

    # 交错合并正弦和余弦
    pos_encoding = tf.reshape(tf.stack([sin_part, cos_part], axis=-1),
                              [1, seq_length, d_model])

    # 添加位置编码
    transformer_input = normalized_input + pos_encoding

    # Multi-Head Attention
    attention_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim
    )(transformer_input, transformer_input)

    # 残差连接和层归一化
    attention_output = Add()([transformer_input, attention_output])
    attention_output = LayerNormalization()(attention_output)

    # 前馈网络
    ff_output = Dense(128, activation='relu')(attention_output)
    ff_output = Dense(d_model)(ff_output)

    # 残差连接和层归一化
    encoder_output = Add()([attention_output, ff_output])
    encoder_output = LayerNormalization()(encoder_output)

    # Dropout
    dropout_output = Dropout(dropout_rate)(encoder_output)

    return dropout_output


def create_balanced_model(input_shape):
    """创建平衡的CNN-Transformer-LSTM模型"""
    inputs = tf.keras.layers.Input(shape=input_shape)

    # CNN block - 提取局部特征
    x = Conv1D(64, kernel_size=7, strides=1, padding="same", activation="relu",
               kernel_initializer="he_normal")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu",
               kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, kernel_size=3, strides=1, padding="same", activation="relu",
               kernel_initializer="he_normal")(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Transformer Encoder Block - 捕获长期依赖
    transformer_output = transformer_encoder_block(x, num_heads=4, key_dim=32, dropout_rate=0.3)

    # LSTM - 处理时序信息
    lstm_output = LSTM(units=128, dropout=0.3, recurrent_dropout=0.3,
                       activation='tanh', return_sequences=False)(transformer_output)

    # Fully Connected Layers
    fc_output = Dense(128, activation='relu')(lstm_output)
    fc_output = Dropout(0.3)(fc_output)

    fc_output = Dense(64, activation='relu')(fc_output)
    fc_output = Dropout(0.3)(fc_output)

    outputs = Dense(2, activation="softmax")(fc_output)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def plot_training_history(history, model, x_test, y_test, groups_test):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    history_dict = history.history

    axes[0].plot(history_dict["loss"], "r-", label="Training Loss", linewidth=0.5)
    axes[0].plot(history_dict["val_loss"], "b-", label="Validation Loss", linewidth=0.5)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history_dict["accuracy"], "r-", label="Training Accuracy", linewidth=0.5)
    axes[1].plot(history_dict["val_accuracy"], "b-", label="Validation Accuracy", linewidth=0.5)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    plt.savefig('train_history.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    # 保存预测分数
    y_score = model.predict(x_test, verbose=1)
    output = pd.DataFrame({
        "y_true": np.argmax(y_test, axis=-1),
        "y_score_apnea": y_score[:, 1],
        "y_score_normal": y_score[:, 0],
        "subject": groups_test
    })
    output.to_csv("CNN-Transformer-LSTM.csv", index=False)

    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(y_score, axis=-1)

    # 计算混淆矩阵
    C = confusion_matrix(y_true, y_pred, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]

    # 计算各项指标
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    sn = TP / (TP + FN) if (TP + FN) > 0 else 0
    sp = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = f1_score(y_true, y_pred, average='binary')

    # 计算AUC
    fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_score[:, 1])
    roc_auc = auc(fpr, tpr)

    # 计算Kappa
    po = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (TP + TN + FP + FN) ** 2 if (TP + TN + FP + FN) > 0 else 0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0

    print("=" * 60)
    print("模型评估结果:")
    print(f"准确率 (Accuracy): {acc:.4f}")
    print(f"灵敏度 (Sensitivity/Recall): {sn:.4f}")
    print(f"特异度 (Specificity): {sp:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"混淆矩阵:")
    print(f"  TP: {TP}, FN: {FN}")
    print(f"  FP: {FP}, TN: {TN}")
    print("=" * 60)

    # 绘制混淆矩阵
    labels = ['Apnea', 'Non-Apnea']
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(C, annot=True, cmap='Reds', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion_Matrix.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 绘制ROC曲线
    fig, ax = plt.subplots()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    plt.savefig('ROC_Curve.png', bbox_inches='tight', dpi=300)
    plt.close()

    return {
        'accuracy': acc,
        'sensitivity': sn,
        'specificity': sp,
        'f1_score': f1,
        'auc': roc_auc,
        'kappa': kappa
    }


if __name__ == "__main__":
    # 加载数据
    print("正在加载数据...")
    x_train, y_train, groups_train, x_test, y_test, groups_test = load_data_with_analysis()

    # 转换为分类格式
    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    print(f"\n训练样本数: {len(y_train)}")
    print(f"测试样本数: {len(y_test)}")
    print(f"输入形状: {x_train.shape[1:]}")

    # 计算类别权重
    y_train_labels = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"\n类别权重: {class_weight_dict}")
    print(f"正常样本权重: {class_weight_dict[0]:.4f}")
    print(f"呼吸暂停样本权重: {class_weight_dict[1]:.4f}")

    # 创建模型
    print("\n正在创建模型...")
    model = create_balanced_model(input_shape=x_train.shape[1:])

    # 显示模型摘要
    model.summary()

    # 绘制模型结构图
    try:
        plot_model(model, "model_architecture.png", show_shapes=True)
        print("模型结构图已保存为 model_architecture.png")
    except Exception as e:
        print(f"无法绘制模型结构图: {e}")

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    # 设置回调函数
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        verbose=1,
        min_lr=1e-6
    )

    csv_logger = tf.keras.callbacks.CSVLogger(
        'training_log.csv',
        separator=',',
        append=False
    )

    callbacks_list = [
        checkpoint,
        early,
        reduce_lr,
        csv_logger,
    ]

    # 训练模型
    print("\n开始训练模型...")
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=100,
        validation_data=(x_test, y_test),
        callbacks=callbacks_list,
        class_weight=class_weight_dict,  # 添加类别权重
        verbose=1
    )

    # 保存最终模型
    model.save("final_model.h5")
    print("最终模型已保存为 final_model.h5")

    # 评估并绘图
    print("\n正在评估模型...")
    results = plot_training_history(history, model, x_test, y_test, groups_test)

    # 保存结果
    with open('model_results.txt', 'w') as f:
        f.write("CNN-Transformer-LSTM 模型结果\n")
        f.write("=" * 40 + "\n")
        for key, value in results.items():
            f.write(f"{key}: {value:.4f}\n")

    print("\n所有任务完成!")
