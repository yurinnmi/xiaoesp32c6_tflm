import argparse
import numpy as np
import tensorflow as tf

def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)  # (N,128,3)
    y = d["y"].astype(np.int64)    # (N,)
    return X, y

def concat_npz(paths):
    Xs, ys = [], []
    for p in paths:
        X, y = load_npz(p)
        Xs.append(X); ys.append(y)
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

def make_model(input_shape=(128,3), num_classes=4):
    # 小型1D-CNN（TFLM向け）
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(16, 5, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool1D(2)(x)  # 64
    x = tf.keras.layers.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool1D(2)(x)  # 32
    x = tf.keras.layers.Conv1D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inp, out)
    return model

def make_datasets(X_train, y_train, X_val, y_val, batch=32):
    ds_tr = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch).prefetch(tf.data.AUTOTUNE)
    ds_va = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds_tr, ds_va

def eval_report(model, X, y, name="test"):
    y_prob = model.predict(X, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    acc = (y_pred == y).mean()
    print(f"[INFO] {name} acc = {acc:.4f}")
    # 混同行列
    cm = tf.math.confusion_matrix(y, y_pred, num_classes=4).numpy()
    print(f"[INFO] {name} confusion matrix:\n{cm}")
    return acc, cm

def convert_tflite_float(model, out_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = []  # float32
    tflite = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite)
    print("[OK] saved:", out_path)

def convert_tflite_int8(model, out_path, rep_data):
    # rep_data: (N,128,3) float32
    def representative_dataset():
        # 代表データは数百例で十分
        for i in range(min(len(rep_data), 300)):
            yield [rep_data[i:i+1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite)
    print("[OK] saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True, help="train npz files (e.g. session_01.npz session_02.npz)")
    ap.add_argument("--test", nargs="+", required=True, help="test npz files (e.g. session_03.npz)")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="split train into train/val by ratio")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    X_tr_all, y_tr_all = concat_npz(args.train)
    X_te, y_te = concat_npz(args.test)

    # train内を val に分割（シャッフルして割合分割）
    idx = np.arange(len(X_tr_all))
    np.random.shuffle(idx)
    X_tr_all = X_tr_all[idx]
    y_tr_all = y_tr_all[idx]

    n_val = int(len(X_tr_all) * args.val_ratio)
    X_val, y_val = X_tr_all[:n_val], y_tr_all[:n_val]
    X_tr,  y_tr  = X_tr_all[n_val:], y_tr_all[n_val:]

    print("[INFO] train:", X_tr.shape, "val:", X_val.shape, "test:", X_te.shape)
    # クラス分布
    for name, y in [("train", y_tr), ("val", y_val), ("test", y_te)]:
        u, c = np.unique(y, return_counts=True)
        print("[INFO]", name, "class counts:", dict(zip(u.tolist(), c.tolist())))

    model = make_model(input_shape=X_tr.shape[1:], num_classes=4)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    model.summary()

    ds_tr, ds_va = make_datasets(X_tr, y_tr, X_val, y_val, batch=args.batch)

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=6, min_lr=1e-5),
    ]

    model.fit(ds_tr, validation_data=ds_va, epochs=args.epochs, callbacks=cb, verbose=2)

    eval_report(model, X_tr, y_tr, name="train")
    eval_report(model, X_te, y_te, name="test")

    # 保存
    model.save("gesture_model.keras")
    print("[OK] saved: gesture_model.keras")

    # TFLite（float32）
    convert_tflite_float(model, "gesture_model_float32.tflite")

    # TFLite（int8）
    # 代表データは train から使う（すでに前処理済みfloat）
    convert_tflite_int8(model, "gesture_model_int8.tflite", X_tr)

if __name__ == "__main__":
    main()
